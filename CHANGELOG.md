# Changelog

## 1.2.0 (2026-02-27)

### Features

- feature-gate gym dependency and migrate to gymnasia for crates.io publishing ([a03866e](https://github.com/urmzd/linear-gp/commit/a03866e3cfc4775ac5b32cfcfbe0016e868f35a0))

### Documentation

- remove License section from README ([361d356](https://github.com/urmzd/linear-gp/commit/361d356b59c163736a354fe8a0c6ce805f93d619))

### Miscellaneous

- standardize GitHub Actions workflows ([ed8234e](https://github.com/urmzd/linear-gp/commit/ed8234eb38ab2cf8638f6d915becd1f9c9d9cfa8))


## 1.1.1 (2026-02-22)

### Bug Fixes

- embed iris dataset locally to avoid network/SSL failures ([f609bea](https://github.com/urmzd/linear-gp/commit/f609beacc9a41f5c4c56a0e48ee6d0e89c8ad210))


## 1.1.0 (2026-02-13)

### Features

- add custom git hook scripts ([5dbe75e](https://github.com/urmzd/linear-gp/commit/5dbe75e93776733d46c634bae6119da271c9ec8d))
- add force re-release support to release workflow ([6057970](https://github.com/urmzd/linear-gp/commit/6057970c2c13184262e8a752330537411e393a01))
- add tracing instrumentation throughout codebase ([7fdb696](https://github.com/urmzd/linear-gp/commit/7fdb696658e0bac76d7a36113d7a2e1c11672b54))
- implement structured tracing and logging infrastructure ([70d0d7c](https://github.com/urmzd/linear-gp/commit/70d0d7c8442648abdd8132874288b6edb73590e1))
- add experiment configuration files for all benchmarks ([90261ec](https://github.com/urmzd/linear-gp/commit/90261ec63aba2fd6cb5e9665a6615c254c9aa718))
- add utility functions for misc operations ([7b5f330](https://github.com/urmzd/linear-gp/commit/7b5f3302eb60a2e9f8109a6f3125c903d11730f4))
- update iris problem solver with new configuration ([9a172c0](https://github.com/urmzd/linear-gp/commit/9a172c098c97d633b51e478d2d1eaf2e1aefa591))
- add Python project configuration with uv support ([f08c155](https://github.com/urmzd/linear-gp/commit/f08c155ebd6393c50e2e7be804935ce3edfd4277))
- add Python tooling package with CLI and utilities ([a504f87](https://github.com/urmzd/linear-gp/commit/a504f874c0cda1951092f72fdf81d0492b71d372))
- add justfile for task automation ([5cc7e82](https://github.com/urmzd/linear-gp/commit/5cc7e829ea3b954ff715617dc2cce5a586b93a5f))
- add iris runner for experiments ([746aabe](https://github.com/urmzd/linear-gp/commit/746aabe9d2be582faba0bbe20fb933a78b4864de))
- add gym runner for experiments ([c27d76d](https://github.com/urmzd/linear-gp/commit/c27d76ddde314cfd1d755e6ade88264d7ea32925))
- add experiments library and main module ([68f29e4](https://github.com/urmzd/linear-gp/commit/68f29e4f8c1df34daf1b147524aa90bda38db9b3))
- add iris baseline experiment outputs ([d8051c1](https://github.com/urmzd/linear-gp/commit/d8051c136dca5d4f7f2b3dfba44ce9192c3c84b0))
- move and reorganize experiment assets and parameters ([85a5992](https://github.com/urmzd/linear-gp/commit/85a5992e512c3606824262ed1b86e85ce59e1925))
- add experiments workspace with Cargo configuration ([24fa7a8](https://github.com/urmzd/linear-gp/commit/24fa7a8272c61520c6936eb0a4d422da5a8b6fd6))
- add iris classification example ([721c68e](https://github.com/urmzd/linear-gp/commit/721c68ebfab6a6cc4b3c9610039ac3032870c2e2))
- add cart pole example ([3f202d6](https://github.com/urmzd/linear-gp/commit/3f202d67dcbfeb914fae989266bfa418c3710c99))
- Add scripts to automate LaTeX compilation and packaging for arXiv submission, update .gitignore, and include all bibliography entries. ([bf6e999](https://github.com/urmzd/linear-gp/commit/bf6e9997252abb3b18ba3026f2f244d129123dd9))

### Bug Fixes

- remove legacy hooks config ([a0ef469](https://github.com/urmzd/linear-gp/commit/a0ef4694289765c3b90aa80919ca353c7c17d5a9))
- resolve CI pipeline failures (SIGILL crash and ruff lint errors) ([8c4f909](https://github.com/urmzd/linear-gp/commit/8c4f909fbca3eb181f3231a5273d178772140b8b))
- adapt gym.rs to current gym-rs fork API ([0dab1a3](https://github.com/urmzd/linear-gp/commit/0dab1a30296a22fd28c9c7b62173fa40023015ee))
- remove python version lock ([5333491](https://github.com/urmzd/linear-gp/commit/5333491c1c1188dc658f11e16cb7d52f496e1a45))

### Documentation

- update extending guide with current registration patterns ([702630b](https://github.com/urmzd/linear-gp/commit/702630b4ba72b39346c080254a9dabe9d51bc129))
- update README with logging and tracing documentation and new output structure ([f8d58e9](https://github.com/urmzd/linear-gp/commit/f8d58e94d521f9b184c3502cfabc40ee17fa6304))
- update contributing guidelines with pre-commit hooks and logging guidelines ([078db52](https://github.com/urmzd/linear-gp/commit/078db52d4260826a580a7886585d8548655810d0))
- update contributing guidelines and extend documentation ([bfbb331](https://github.com/urmzd/linear-gp/commit/bfbb33196785cfe4305bb407230c53852896d60b))
- add extending documentation with comprehensive framework guide ([7f18f1b](https://github.com/urmzd/linear-gp/commit/7f18f1b260ff34f1e2a5760efe08a20f496d2a9a))
- remove .gitignore from docs directory ([f2a68ef](https://github.com/urmzd/linear-gp/commit/f2a68ef2b5aad22ab8a8c8eef0d45cd492112845))
- clarify extending documentation with quick start reference ([aa17a9f](https://github.com/urmzd/linear-gp/commit/aa17a9f9bb9c443e75c7db59f7581cff7aebfd15))
- update README with new structure and setup instructions ([cbb4c37](https://github.com/urmzd/linear-gp/commit/cbb4c3799da48c9efe6e27372b97e88410e5497c))
- add comprehensive contributing guidelines ([7e59733](https://github.com/urmzd/linear-gp/commit/7e597339003baeadce1efa9db6425b126cc836a0))

### Refactoring

- standardize justfile recipe names for better consistency ([8d4ea9f](https://github.com/urmzd/linear-gp/commit/8d4ea9fa7dbd2bd36e56db708de9652d6fa616ca))
- update source files for experiment reorganization ([d602b7b](https://github.com/urmzd/linear-gp/commit/d602b7bd71f8ce398e0ad1711dceffed188f215d))
- update core configuration with new experiment support ([e1df7e7](https://github.com/urmzd/linear-gp/commit/e1df7e7bc1db1601b35c00fbf466ec63b69711b5))
- migrate scripts to lgp_tools package and remove legacy scripts ([20f886d](https://github.com/urmzd/linear-gp/commit/20f886d2b2643a7a4d2a16836a87e64beea555c3))
- move benchmark_tools to experiments workspace ([f71e5b1](https://github.com/urmzd/linear-gp/commit/f71e5b15c1569462fef28e1735da7111a8ac118f))

### Miscellaneous

- update Cargo.lock ([63eb1eb](https://github.com/urmzd/linear-gp/commit/63eb1eb78b8d0c3e7ac6b73810b8714dea538249))
- update justfile to use custom git hooks ([480b782](https://github.com/urmzd/linear-gp/commit/480b782c4504c54d2b9443d24425b52b2b650aad))
- remove pre-commit framework and simd-json dependency ([6bcca66](https://github.com/urmzd/linear-gp/commit/6bcca660ff60694eb39790426d43e2ce51d9a031))
- update semantic-release action to v1 ([dfd137e](https://github.com/urmzd/linear-gp/commit/dfd137eb49aacbca8421d3454082dc1d95023763))
- license under Apache 2.0 ([f9cac83](https://github.com/urmzd/linear-gp/commit/f9cac839ea756dbe4e82d43a661b387ccdaec02b))
- remove crates.io publish step from release workflow ([69aa1b6](https://github.com/urmzd/linear-gp/commit/69aa1b652fd9b652b7a01d487f7524a7195a5140))
- update local settings with additional bash permissions ([4c87ad4](https://github.com/urmzd/linear-gp/commit/4c87ad44b6c50ed0574d09a44462c28fec45b77a))
- add just recipes for verbose and trace logging ([e25f80f](https://github.com/urmzd/linear-gp/commit/e25f80f814856e2dfa2e45595f283cd8dd8055fd))
- update gitignore to exclude outputs and log files ([92a8b26](https://github.com/urmzd/linear-gp/commit/92a8b26d278f91dc51f222093a46195112a01a60))
- add dependency installation step to experiments workflow ([fce031e](https://github.com/urmzd/linear-gp/commit/fce031e9703afb2b0e1aa854bdfbb3c2101705d2))
- add --dev flag to uv sync ([b26501a](https://github.com/urmzd/linear-gp/commit/b26501afe5dff84ad9ee44bfa319348f47b65782))
- cleanup ([db08779](https://github.com/urmzd/linear-gp/commit/db087796f6de2fde9a8eb1986ca666d15341d646))
- add local settings and pre-commit configuration ([bf6b4c1](https://github.com/urmzd/linear-gp/commit/bf6b4c16a1dd357b538d45623b55897c70ee477a))
- update project metadata and configuration ([fcfb518](https://github.com/urmzd/linear-gp/commit/fcfb518970e8ae489ee6d9752185ffe76148f878))
- migrate to workspace structure with lgp and lgp-cli crates ([fec89fb](https://github.com/urmzd/linear-gp/commit/fec89fbe60a2d58ad7006064d47dedaa75a194af))
- migrate to uv_build backend with module root configuration ([b351378](https://github.com/urmzd/linear-gp/commit/b351378c21dea46e255419cac45d732aa92d1371))
- format code with proper line breaks and imports ([a336c4f](https://github.com/urmzd/linear-gp/commit/a336c4fd5b4df96bc0d97097791bd0033c50b1f9))
- refactor experiments workflow with input parameters and improved structure ([c4483cd](https://github.com/urmzd/linear-gp/commit/c4483cd44cbcb95c4ce2f992b557abad50036e9c))
- update workflow action versions and add arXiv submission job ([515ab8c](https://github.com/urmzd/linear-gp/commit/515ab8c76b937b2207766ada5df71e76962c2569))
- add comprehensive CI workflow with linting and tests ([1927867](https://github.com/urmzd/linear-gp/commit/1927867daad89f64af99e3e122dac84941a8fbf4))
- cleanup legacy experimentation results ([09cb80d](https://github.com/urmzd/linear-gp/commit/09cb80dbe493713083f56fdf6afa9d547a4826a7))
- add comprehensive smoke tests ([c4b6fa2](https://github.com/urmzd/linear-gp/commit/c4b6fa24f4a112a7bc5b6c7cd8a3a72f01f33aca))
- update module exports ([d9bdd44](https://github.com/urmzd/linear-gp/commit/d9bdd44ed6e29e89717792efba661564bba3798a))
- remove test utilities module ([c5f2cac](https://github.com/urmzd/linear-gp/commit/c5f2cac1c06ce12797e7aa744f76bf423e6bcda2))
- remove gym problem solver from main package ([bfdf224](https://github.com/urmzd/linear-gp/commit/bfdf224a1f7018fe7a68be9c3f4d26393e0911fb))
- update Cargo dependencies ([d21e01d](https://github.com/urmzd/linear-gp/commit/d21e01df3fc462bc7f853ec3949a61b166b15a31))
- add Python version specification ([0319ad1](https://github.com/urmzd/linear-gp/commit/0319ad11fc0a1ea630900b8326b9bef39869db78))
- update .gitignore with Python and UV dependencies ([7506103](https://github.com/urmzd/linear-gp/commit/7506103269efe0950a1319b67db598004d230dfd))
- rename .cargo/config to .cargo/config.toml ([bedad42](https://github.com/urmzd/linear-gp/commit/bedad422e26697aa79fe4a1ca70dcf106e579d0f))
- cleanup files ([a3a46fe](https://github.com/urmzd/linear-gp/commit/a3a46fea19cc10f9e7971a0bb63b81451848cd52))
