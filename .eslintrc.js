module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true
  },
  extends: [
    'eslint:recommended'
  ],
  parserOptions: {
    ecmaVersion: 12,
    sourceType: 'module'
  },
  rules: {
    'indent': ['error', 2],
    'linebreak-style': ['error', 'unix'],
    'quotes': ['error', 'single'],
    'semi': ['error', 'always'],
    'no-unused-vars': ['warn'],
    'no-console': 'off',
    'comma-dangle': ['error', 'never'],
    'arrow-parens': ['error', 'always']
  },
  globals: {
    'window': true,
    'document': true,
    'process': true,
    '__dirname': true,
    'require': true,
    'module': true,
    'exports': true
  }
};