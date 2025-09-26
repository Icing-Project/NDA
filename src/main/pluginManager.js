const EventEmitter = require('events');
const path = require('path');
const fs = require('fs').promises;

class PluginManager extends EventEmitter {
    constructor() {
        super();
        this.plugins = new Map();
        this.activePlugins = new Set();
        this.pluginDirectory = path.join(__dirname, '../plugins');
    }

    async loadPlugins() {
        console.log('Loading plugins from:', this.pluginDirectory);

        try {
            // Load input plugins
            await this.loadPluginType('inputs');

            // Load processor plugins
            await this.loadPluginType('processors');

            // Load output plugins
            await this.loadPluginType('outputs');

            console.log(`Loaded ${this.plugins.size} plugins`);
            return Array.from(this.plugins.values());
        } catch (error) {
            console.error('Failed to load plugins:', error);
            return [];
        }
    }

    async loadPluginType(type) {
        const typePath = path.join(this.pluginDirectory, type);

        try {
            const files = await fs.readdir(typePath);
            const jsFiles = files.filter(file => file.endsWith('.js'));

            for (const file of jsFiles) {
                await this.loadPlugin(path.join(typePath, file), type);
            }
        } catch (error) {
            console.warn(`No ${type} plugins found or directory doesn't exist`);
        }
    }

    async loadPlugin(pluginPath, type) {
        try {
            // Clear require cache for hot reloading
            delete require.cache[require.resolve(pluginPath)];

            const PluginClass = require(pluginPath);
            const pluginInstance = new PluginClass();

            // Validate plugin
            if (!this.validatePlugin(pluginInstance)) {
                console.warn(`Invalid plugin at ${pluginPath}`);
                return;
            }

            const pluginInfo = {
                id: pluginInstance.id || path.basename(pluginPath, '.js'),
                name: pluginInstance.name,
                description: pluginInstance.description,
                version: pluginInstance.version || '1.0.0',
                type: type,
                path: pluginPath,
                instance: pluginInstance,
                active: false
            };

            this.plugins.set(pluginInfo.id, pluginInfo);
            console.log(`Loaded plugin: ${pluginInfo.name} (${pluginInfo.id})`);

        } catch (error) {
            console.error(`Failed to load plugin from ${pluginPath}:`, error);
        }
    }

    validatePlugin(plugin) {
        // Check required properties and methods
        const requiredProperties = ['name', 'description'];
        const requiredMethods = ['initialize', 'cleanup'];

        for (const prop of requiredProperties) {
            if (!plugin[prop]) {
                console.warn(`Plugin missing required property: ${prop}`);
                return false;
            }
        }

        for (const method of requiredMethods) {
            if (typeof plugin[method] !== 'function') {
                console.warn(`Plugin missing required method: ${method}`);
                return false;
            }
        }

        return true;
    }

    getAvailablePlugins() {
        return Array.from(this.plugins.values()).map(plugin => ({
            id: plugin.id,
            name: plugin.name,
            description: plugin.description,
            version: plugin.version,
            type: plugin.type,
            active: plugin.active
        }));
    }

    async activatePlugin(pluginId) {
        const plugin = this.plugins.get(pluginId);

        if (!plugin) {
            throw new Error(`Plugin ${pluginId} not found`);
        }

        if (plugin.active) {
            return { success: true, message: 'Plugin already active' };
        }

        try {
            await plugin.instance.initialize();
            plugin.active = true;
            this.activePlugins.add(pluginId);

            // Set up event forwarding
            this.setupPluginEventForwarding(plugin);

            this.emit('pluginActivated', {
                id: pluginId,
                name: plugin.name
            });

            console.log(`Activated plugin: ${plugin.name}`);
            return { success: true, message: `Plugin ${plugin.name} activated` };

        } catch (error) {
            console.error(`Failed to activate plugin ${pluginId}:`, error);
            throw error;
        }
    }

    async deactivatePlugin(pluginId) {
        const plugin = this.plugins.get(pluginId);

        if (!plugin) {
            throw new Error(`Plugin ${pluginId} not found`);
        }

        if (!plugin.active) {
            return { success: true, message: 'Plugin already inactive' };
        }

        try {
            await plugin.instance.cleanup();
            plugin.active = false;
            this.activePlugins.delete(pluginId);

            // Remove event forwarding
            this.removePluginEventForwarding(plugin);

            this.emit('pluginDeactivated', {
                id: pluginId,
                name: plugin.name
            });

            console.log(`Deactivated plugin: ${plugin.name}`);
            return { success: true, message: `Plugin ${plugin.name} deactivated` };

        } catch (error) {
            console.error(`Failed to deactivate plugin ${pluginId}:`, error);
            throw error;
        }
    }

    setupPluginEventForwarding(plugin) {
        // Forward plugin events to main event bus
        const eventHandlers = new Map();

        const events = ['data', 'error', 'status', 'warning'];
        events.forEach(eventName => {
            const handler = (...args) => {
                this.emit(`plugin:${eventName}`, {
                    pluginId: plugin.id,
                    pluginName: plugin.name,
                    data: args
                });
            };

            plugin.instance.on(eventName, handler);
            eventHandlers.set(eventName, handler);
        });

        // Store handlers for cleanup
        plugin._eventHandlers = eventHandlers;
    }

    removePluginEventForwarding(plugin) {
        if (plugin._eventHandlers) {
            plugin._eventHandlers.forEach((handler, eventName) => {
                plugin.instance.removeListener(eventName, handler);
            });
            delete plugin._eventHandlers;
        }
    }

    async reloadPlugins() {
        console.log('Reloading all plugins...');

        // Deactivate all active plugins
        const activePluginIds = Array.from(this.activePlugins);
        for (const pluginId of activePluginIds) {
            await this.deactivatePlugin(pluginId);
        }

        // Clear plugin registry
        this.plugins.clear();

        // Reload all plugins
        await this.loadPlugins();

        // Reactivate previously active plugins if they still exist
        for (const pluginId of activePluginIds) {
            if (this.plugins.has(pluginId)) {
                try {
                    await this.activatePlugin(pluginId);
                } catch (error) {
                    console.error(`Failed to reactivate plugin ${pluginId}:`, error);
                }
            }
        }

        this.emit('pluginsReloaded');
        return this.getAvailablePlugins();
    }

    getActivePlugins() {
        return Array.from(this.activePlugins).map(pluginId => {
            const plugin = this.plugins.get(pluginId);
            return {
                id: plugin.id,
                name: plugin.name,
                instance: plugin.instance
            };
        });
    }

    getPluginInstance(pluginId) {
        const plugin = this.plugins.get(pluginId);
        return plugin ? plugin.instance : null;
    }

    async cleanup() {
        console.log('Cleaning up plugin manager...');

        // Deactivate all active plugins
        for (const pluginId of this.activePlugins) {
            try {
                await this.deactivatePlugin(pluginId);
            } catch (error) {
                console.error(`Error deactivating plugin ${pluginId} during cleanup:`, error);
            }
        }

        this.removeAllListeners();
        this.plugins.clear();
        this.activePlugins.clear();
    }
}

module.exports = { PluginManager };