# Agent Planner Enhancement Prompt

## Task
Review and enhance the NDA v2.0 migration plan (`nda_v2.0_migration_008441a9.plan.md`) to make it production-ready and implementation-complete. The plan currently covers core technical migration but needs additional detail, risk considerations, and comprehensive validation criteria.

## Critical Enhancement Areas

### 3. Risk Mitigation
**Add a dedicated section** identifying potential failure modes for each major component:

- **Resampler**: What if quality is insufficient? What if libsamplerate isn't available?
- **Python Bridge**: What if optimization fails to meet <500µs target?
- **Dual Pipeline UI**: What if UI becomes too complex? What if users configure incorrectly?
- **Crypto Plugins**: What if key exchange fails? What if plugins crash?

For each risk, provide:
- **Risk**: Clear description of what could go wrong
- **Impact**: Severity (High/Medium/Low)
- **Mitigation**: Proactive prevention steps
- **Fallback**: Plan B if mitigation fails

### 4. Success Metrics
**Expand Success Criteria** with measurable, testable metrics:

- **Code Quality**: Add specific targets (e.g., "ProcessingPipeline.cpp < 500 lines", "Zero compiler warnings")
- **Performance**: Quantify with units (e.g., "Python bridge < 500µs per buffer", "CPU < 30%")
- **Stability**: Define test scenarios (e.g., "24-hour soak test", "1000 start/stop cycles")
- **Usability**: Add user-facing criteria (e.g., "One-click 'Start Both' works", "Clear error messages")

Include validation commands (e.g., `grep -r "bearer"` to verify removal).

### 6. Resampler Implementation
**Enhance Phase 3** with complete specifications:

- Add `Medium` quality option (not just Simple/High)
- Include destructor `~Resampler()` in header
- Document `lastSamples_` usage for buffer continuity
- Add complete libsamplerate CMake configuration
- Specify test cases: 44.1→48, 96→48, pass-through, multi-channel
- Document how to handle buffer boundaries (avoid clicks/pops)

### 7. Python Bridge Optimization
**Enhance Phase 4** with quantified performance targets:

- Document baseline: 3000-15000µs (before optimization)
- Break down per-step improvements:
  - Object caching: 3000µs → 1500µs (50% reduction)
  - Zero-copy: 1500µs → 500µs (67% reduction)
  - Batch GIL: 500µs → 300µs (40% reduction)
- Include complete benchmark code for BOTH source AND processor plugins
- Add validation checklist for each optimization step
- Document how to measure GIL contention

### 8. Example Plugins
**Enhance Phase 7** with complete, production-ready examples:

- **Add Passthrough Processor**: Testing utility plugin (`plugins_py/examples/passthrough.py`)
- **Full lifecycle**: Include initialize, start, stop, shutdown for ALL examples
- **Parameter handling**: Add setParameter/getParameter examples
- **Plugin exports**: Include `extern "C"` export functions for C++ plugins
- **Error handling**: Show proper error handling patterns
- **Memory management**: Document cleanup (destructors, resource freeing)

### 9. UI Implementation
**Enhance Phase 6** with complete widget specifications:

- **Complete widget list**: Every combo box, button, label, status indicator
- **Signal/slot connections**: Document ALL Qt connections
- **State management**: Define UI state transitions (Not configured → Running → Error)
- **Status indicators**: Specify text/colors/icons (e.g., "⚙️ Not configured", "🟢 TX Running", "🔴 TX Error")
- **Error display**: How are errors shown to users?
- **Slot implementations**: Include complete `onStartTxClicked()`, `onStartBothClicked()` examples

### 10. Metrics Implementation
**Enhance Phase 5** with complete metric specifications:

- **All metric getters**: List every getter (including `getDriftWarnings()`, `getBackpressureWaits()`)
- **Calculation formulas**: Document how each metric is computed
- **Drift warning logic**: Log every 100th warning, format: `[Pipeline] Warning: 125ms behind schedule`
- **Update frequency**: Dashboard refreshes every 100ms
- **Backpressure retry logic**: Wait 5ms, retry once, then drop

### 11. CMakeLists.txt Updates
**Add explicit CMake change specifications:**

- **Crypto removal**: Show exact lines to remove from core
- **Examples subdirectory**: Create `plugins_src/examples/CMakeLists.txt` with plugin builds
- **libsamplerate**: Show optional dependency detection with `find_package(PkgConfig)`
- **Install layout**: Document plugin installation paths

### 14. Code Examples
**Ensure all code examples are complete and production-ready:**

- **Full class definitions**: Constructors, destructors, all methods
- **Error handling**: Proper error checking and handling
- **Memory management**: RAII patterns, cleanup in destructors
- **Thread safety**: Note thread-safety considerations
- **Comments**: Explanatory comments for complex logic
- **Compilability**: Examples must actually compile

## Enhancement Process

1. **Review each phase** of the original plan
2. **Identify gaps** using the criteria above
3. **Add missing details** following the examples
4. **Ensure consistency** across all sections
5. **Add validation criteria** for each enhancement
6. **Include testing requirements** where applicable

## Output Requirements

The enhanced plan should:
- Be **implementation-ready** - developers can follow step-by-step without clarification
- Include **complete code examples** - copy-paste ready
- Have **measurable success criteria** - testable validation
- Document **risks and mitigations** - proactive problem-solving
- Provide **validation commands** - how to verify each step

## Questions to Guide Enhancement

For each section, ask:
- **What could go wrong?** → Add to Risk Mitigation
- **How do we measure success?** → Add to Success Metrics
- **What's missing from the code example?** → Complete the example
- **How do we test this?** → Add testing requirements
- **What if this fails?** → Add fallback plan

## Success Criteria for Enhancement

The enhanced plan is complete when:
- ✅ Every code example compiles and runs
- ✅ Every metric has a validation method
- ✅ Every risk has a mitigation strategy
- ✅ Every phase has testing requirements
- ✅ A developer can implement without asking questions

