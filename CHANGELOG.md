# Changelog

## 1.7.10 (2026-04-19)

### Bug Fixes

- retrigger release with sr v7.1.2 Cargo.lock handling ([84943ca](https://github.com/urmzd/linear-gp/commit/84943cad87b4ddf75e85125791cf9a277695871a))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.9...v1.7.10)


## 1.7.9 (2026-04-19)

### Bug Fixes

- **ci**: apply cargo fmt, sync lockfile, stage Cargo.lock ([b209ac2](https://github.com/urmzd/linear-gp/commit/b209ac2d33c6c57ca5b22f84e3205c132615e699))
- **ci**: drop removed sr force input and nonexistent init --merge flag ([37b8762](https://github.com/urmzd/linear-gp/commit/37b876217282f3e728df35f9a63e6a6753164465))

### Refactoring

- move cargo publish into sr hooks.post_release ([a3f6c57](https://github.com/urmzd/linear-gp/commit/a3f6c57fcca98a10648a86c528863defabc5ce98))

### Misc

- revamp showcase with operator comparison and benchmarks ([a4dc958](https://github.com/urmzd/linear-gp/commit/a4dc958c8dbaec62edd7590c3fab0f091fe19b5e))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.8...v1.7.9)


## 1.7.8 (2026-04-16)

### Bug Fixes

- update README install commands to use sh instead of bash ([e8b8caa](https://github.com/urmzd/linear-gp/commit/e8b8caafe69eb2ed184341c51f6566bab4d5698d))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.7...v1.7.8)


## 1.7.7 (2026-04-16)

### Bug Fixes

- install to ~/.local/bin by default and add PATH setup ([d911fdc](https://github.com/urmzd/linear-gp/commit/d911fdce36885f6c88b6e2558a02559ac6d22b1a))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.6...v1.7.7)


## 1.7.6 (2026-04-16)

### Bug Fixes

- **ci**: migrate sr v4 to v7 for artifact and input support (#20) ([3b59962](https://github.com/urmzd/linear-gp/commit/3b599629f6b7e7e5a381efe91392b3a3f79395d8))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.5...v1.7.6)


## 1.7.5 (2026-04-15)

### Refactoring

- inline agentspec-update (#18) ([3759137](https://github.com/urmzd/linear-gp/commit/37591372f477862b54281d762b0c2647152e1918))

### Miscellaneous

- migrate sr config and action to v4 ([c8f751a](https://github.com/urmzd/linear-gp/commit/c8f751a64899bcc8e5f88314e74b785711a36ea8))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.4...v1.7.5)


## 1.7.4 (2026-04-10)

### Bug Fixes

- **ci**: publish lgp-core before lgp to crates.io ([0125cdd](https://github.com/urmzd/linear-gp/commit/0125cddf915c7e73647e9070b697bdce537fd19b))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.3...v1.7.4)


## 1.7.3 (2026-04-10)

### Refactoring

- migrate gymnasia dependency from v1 to v3 ([56497b2](https://github.com/urmzd/linear-gp/commit/56497b25966d850390c500f465f906a3154a7864))

### Miscellaneous

- **benchmarks**: update performance benchmark for new experiment output structure ([33e8f88](https://github.com/urmzd/linear-gp/commit/33e8f880a0c48e3ed7fabb84b113825f5e9738eb))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.2...v1.7.3)


## 1.7.2 (2026-04-09)

### Bug Fixes

- **ci**: remove --allow-dirty from cargo publish ([67175ff](https://github.com/urmzd/linear-gp/commit/67175ff7e13f0e6ea8658e972350940b059736d7))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.1...v1.7.2)


## 1.7.1 (2026-04-09)

### Bug Fixes

- **ci**: checkout release tag, deduplicate publish, handle already-published ([b35a3b3](https://github.com/urmzd/linear-gp/commit/b35a3b30f87154adc297640982e1a06976034c89))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.7.0...v1.7.1)


## 1.7.0 (2026-04-09)

### Features

- **cli**: add self-update, version, and --format flag ([0dc9de1](https://github.com/urmzd/linear-gp/commit/0dc9de1ded6951758e41aca1c6da0779dd922e45))

### Documentation

- add LICENSE to sub-crates for publishing compliance ([eadc79f](https://github.com/urmzd/linear-gp/commit/eadc79f05bdfbae565e7fc1e55bbedaf9de83443))
- fix incorrect crate names, versions, and non-existent CLI subcommand ([6bc227c](https://github.com/urmzd/linear-gp/commit/6bc227c5c48d5e2e29543ffcc905505ad181cdc7))
- update legacy lgp-cli references to lgp ([1f77fcf](https://github.com/urmzd/linear-gp/commit/1f77fcfa2c8dcb7d70228b5978c765c68c764da4))

### Miscellaneous

- add linguist overrides to fix language stats (#16) ([0093d2a](https://github.com/urmzd/linear-gp/commit/0093d2a946fdec91bc277ed918e9f07b290cfe30))
- **deps**: bump actions/create-github-app-token from 1 to 3 ([12b8e5b](https://github.com/urmzd/linear-gp/commit/12b8e5b45b5e93df3b87cc12ca47eb87ec9144db))
- update sr action from v2 to v3 ([75e9ef2](https://github.com/urmzd/linear-gp/commit/75e9ef2ec0c48d457c391608cc8d09466d372010))
- standardize sr.yaml and justfile — floating_tags, refactor bump, full recipes ([3325d36](https://github.com/urmzd/linear-gp/commit/3325d3655362619671a530c605bd6d240c2e262a))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.6.1...v1.7.0)


## 1.6.1 (2026-03-29)

### Bug Fixes

- add examples ([11bbcc7](https://github.com/urmzd/linear-gp/commit/11bbcc744c633de389c54a2fc8c8519961935442))

### Documentation

- **readme**: update example usage instructions ([4600b73](https://github.com/urmzd/linear-gp/commit/4600b734419625ba118b87c5e2ca4b69cc95c0d6))
- restore overview, CLI reference, search, and output sections ([8d179c7](https://github.com/urmzd/linear-gp/commit/8d179c7788ca9116d170b77ab56aa5265e97d4e7))
- move references to rlgp-thesis repository ([8883ee0](https://github.com/urmzd/linear-gp/commit/8883ee0ab5fec88db28cd48ef944e8ff30ceb2a9))
- restore full references section ([31e6657](https://github.com/urmzd/linear-gp/commit/31e6657ea605a0a94d0812fe7c0322ab10c6fbcf))
- simplify README to focus on install.sh and core usage ([380d69e](https://github.com/urmzd/linear-gp/commit/380d69e35f50dbdf2aa21d75579a00326d26ffe9))
- update documentation for simplified setup ([130cb30](https://github.com/urmzd/linear-gp/commit/130cb30bd865c0fe5394caeaad56e149f2544142))

### Refactoring

- **examples**: move examples to lgp crate directory ([94850df](https://github.com/urmzd/linear-gp/commit/94850df56df13ce636c29cab42335cffef215579))
- **cli**: remove example command ([9466be3](https://github.com/urmzd/linear-gp/commit/9466be3f1d3c4eec82f0ca35ebb73855a376886e))
- rename lgp-cli to lgp and lgp to lgp-core for simpler cargo install ([73ec7a0](https://github.com/urmzd/linear-gp/commit/73ec7a00b302e86ccd3b7d5a5c674592594fde00))

### Miscellaneous

- **install**: add installation script for prebuilt binaries ([6be0809](https://github.com/urmzd/linear-gp/commit/6be08098863e8232e8bea184b52fe1151a45f7ad))
- **build**: simplify justfile to core commands ([f5de875](https://github.com/urmzd/linear-gp/commit/f5de87584eab51a142d651a1604a47ca55b3dd68))
- **release**: add cross-platform binary builds ([55a8ebf](https://github.com/urmzd/linear-gp/commit/55a8ebfa36611c5687fed42ebd21e61d75fbd6e3))
- remove obsolete latex compilation script ([4f46759](https://github.com/urmzd/linear-gp/commit/4f46759f252b14ccc8636bfa71c55c8275ccbac9))
- **hooks**: migrate git hooks to sr management ([26b3aaa](https://github.com/urmzd/linear-gp/commit/26b3aaad3b15f961119a03fe537b8b01be697a03))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.6.0...v1.6.1)


## 1.6.0 (2026-03-29)

### Features

- **core**: expose n_threads CLI flag for parallel evaluation ([5c5a2f4](https://github.com/urmzd/linear-gp/commit/5c5a2f4b883b859d10649b059e488ba2e9076197))
- **iris**: derive Clone for IrisState ([19a4b85](https://github.com/urmzd/linear-gp/commit/19a4b857a0872200678d25ecc0cd918c1d6e136e))
- **gym**: add Send + Sync bounds to environment traits ([a0e02b1](https://github.com/urmzd/linear-gp/commit/a0e02b128b0a0683b245c8798dcdfd6fb01e7605))
- **core**: parallelize fitness evaluation with rayon ([8df4dce](https://github.com/urmzd/linear-gp/commit/8df4dcec1b2f088fb205ed1fed3e5473cbe5fe4c))

### Bug Fixes

- **tracing**: use non-blocking stdout to prevent debug logging from stalling computation ([f1e7d5f](https://github.com/urmzd/linear-gp/commit/f1e7d5fa9de978ed587b714374ecc45e423370e9))

### Documentation

- update README ([9651db3](https://github.com/urmzd/linear-gp/commit/9651db39587e14354e0e059647907bcfcf5c52bc))
- **skills**: align SKILL.md with agentskills.io spec ([2d718ae](https://github.com/urmzd/linear-gp/commit/2d718aeb0b942bd68308ba4fbed4785329d51cad))
- update thesis repo link after rename to rlgp-thesis ([25cbc29](https://github.com/urmzd/linear-gp/commit/25cbc29ddd4dbf3b7fe9ebe2236594b8fc821ce5))
- add showcase screenshot ([d74b437](https://github.com/urmzd/linear-gp/commit/d74b4374be070a4531df2f45f315f00257c43539))
- add showcase section to README ([3d20d19](https://github.com/urmzd/linear-gp/commit/3d20d199707d112103a760b8e8796f143011ce53))

### Refactoring

- **benchmark**: use immutable trials reference ([dc2df07](https://github.com/urmzd/linear-gp/commit/dc2df07f61639c2c17c3e26d5aafc662a5afedcb))

### Miscellaneous

- fix rustfmt formatting ([0568a99](https://github.com/urmzd/linear-gp/commit/0568a99a907dfdc05cf0e784b8d1fab5f27c4eaa))
- **benchmark**: add parallel vs sequential fitness evaluation benchmark ([a3986a8](https://github.com/urmzd/linear-gp/commit/a3986a8422dc2c94dc40a94de7fe7f4d70c2e1c9))
- use sr-releaser GitHub App for release workflow (#7) ([b7cce78](https://github.com/urmzd/linear-gp/commit/b7cce78ccbaa0de6fc6ce8ec39fc456125ee6a87))
- update semantic-release action to sr@v2 ([82ffaf6](https://github.com/urmzd/linear-gp/commit/82ffaf606851fc90a93179db56a8c3456dda13c8))
- move thesis to separate repository ([77b5ad3](https://github.com/urmzd/linear-gp/commit/77b5ad3846b55f5d99b3049e49bd9f34d5484a70))
- **teasr**: restructure demo configuration with interactions ([7ad2ac0](https://github.com/urmzd/linear-gp/commit/7ad2ac09953bc11e2036c7cd0403377540722af4))
- **git**: add commit message hooks ([e26a1c5](https://github.com/urmzd/linear-gp/commit/e26a1c52cd52435e912810d934d3240107816517))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.5.0...v1.6.0)


## 1.5.0 (2026-03-21)

### Features

- **cli**: add styled terminal output matching sr UI standard ([3ad8f07](https://github.com/urmzd/linear-gp/commit/3ad8f07e4861944f8aa01d5a074af6d9b173be42))

### Documentation

- consolidate extension guide into skill and rename docs/ to thesis/ ([8562b11](https://github.com/urmzd/linear-gp/commit/8562b1159c430491d3922b4e18fd32dca5fe3a63))

### Miscellaneous

- standardize project files and README header ([3aaa917](https://github.com/urmzd/linear-gp/commit/3aaa917ce2ee410f045461e5039dd22fe38f3474))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.4.1...v1.5.0)


## 1.4.1 (2026-03-14)

### Bug Fixes

- **ci**: remove search phase and reduce log level in experiments ([4808a8b](https://github.com/urmzd/linear-gp/commit/4808a8bbf0d11b14845d06f8ce639677c5af9860))

### Documentation

- expand references to include all cited works ([d832b4d](https://github.com/urmzd/linear-gp/commit/d832b4d4f200cfaca1be5d8faeba15e3dc755e8d))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.4.0...v1.4.1)


## 1.4.0 (2026-03-14)

### Features

- port Python CLI to pure Rust, remove Python ecosystem ([cea926a](https://github.com/urmzd/linear-gp/commit/cea926a934f15d023cb1267c0bfc10346656d28f))

### Documentation

- add AGENTS.md and agent skill for Claude Code ([2433768](https://github.com/urmzd/linear-gp/commit/2433768bde3cb0bc25df30af5f37f46e3c2b4438))

### Miscellaneous

- replace MIT license with Apache-2.0 ([34784a8](https://github.com/urmzd/linear-gp/commit/34784a8cf7a981f9083a5e7b0b17754d97780e0e))
- trigger experiments workflow after each release ([2f4508b](https://github.com/urmzd/linear-gp/commit/2f4508ba4ced570226e0926d8449bd34192a26df))

[Full Changelog](https://github.com/urmzd/linear-gp/compare/v1.3.1...v1.4.0)


## 1.3.1 (2026-02-27)

### Bug Fixes

- skip crates.io publish when no release is created ([9c403bf](https://github.com/urmzd/linear-gp/commit/9c403bf3be6ff6d402929bd432a48b90412d1798))

### Documentation

- add per-package READMEs and link from top-level README ([2242111](https://github.com/urmzd/linear-gp/commit/2242111a97f03de76f5bae5c30bf528039bf2ae8))

### Miscellaneous

- lock ([f86236a](https://github.com/urmzd/linear-gp/commit/f86236ad6207e16ba47f9f79db9112c2e7f01344))
- lock ([fa5f433](https://github.com/urmzd/linear-gp/commit/fa5f433fc4e7badefd57e43f469f754bb9b1bc9a))


## 1.3.0 (2026-02-27)

### Features

- publish lgp-cli to crates.io alongside lgp ([4d44799](https://github.com/urmzd/linear-gp/commit/4d447993cf50d70c5462acad89763d91416640ed))

### Bug Fixes

- use rust-lang/crates-io-auth-action for trusted publishing ([3754c1f](https://github.com/urmzd/linear-gp/commit/3754c1faa6cbf6f0bf87749c83ceaf86e220bf8f))


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
