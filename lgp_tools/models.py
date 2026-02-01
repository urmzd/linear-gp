"""Pydantic models for parsing Rust output and experiment configuration."""

from __future__ import annotations

import json
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, Field

# =============================================================================
# Experiment Configuration Models (for TOML configs)
# =============================================================================


class Metadata(BaseModel):
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    title: str = Field(default="")  # Title for figures/tables
    x_label: str = Field(default="Generation")  # X-axis label
    y_label: str = Field(default="Fitness")  # Y-axis label
    tags: list[str] = Field(default_factory=list)  # For categorization/filtering


class Problem(BaseModel):
    n_inputs: int = Field(default=1)
    n_actions: int = Field(default=1)


class ProgramHyperparameters(BaseModel):
    max_instructions: int = Field(default=100)
    n_extras: int = Field(default=1)
    external_factor: float = Field(default=10.0)


class Hyperparameters(BaseModel):
    population_size: int = Field(default=100)
    n_generations: int = Field(default=200)
    n_trials: int = Field(default=1)
    gap: float = Field(default=0.5)
    default_fitness: float = Field(default=0.0)
    seed: int | None = Field(default=None)
    program: ProgramHyperparameters = Field(default_factory=lambda: ProgramHyperparameters())


class Operation(BaseModel):
    name: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str = Field(default="")
    environment: str = Field(default="")
    metadata: Metadata = Field(default_factory=lambda: Metadata())
    problem: Problem = Field(default_factory=lambda: Problem())
    hyperparameters: Hyperparameters = Field(default_factory=lambda: Hyperparameters())
    operations: list[Operation] = Field(default_factory=list)

    @classmethod
    def from_toml(cls, path: Path) -> ExperimentConfig:
        """Load config from TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    def to_toml(self, path: Path) -> None:
        """Write config to TOML file."""
        with open(path, "wb") as f:
            tomli_w.dump(self.model_dump(exclude_none=True), f)


# =============================================================================
# Rust Output Models (for parsing JSON output)
# =============================================================================


class InstructionGeneratorParameters(BaseModel):
    n_extras: int = Field(default=1)
    external_factor: float = Field(default=10.0)
    n_actions: int = Field(default=1)
    n_inputs: int = Field(default=1)


class ProgramParameters(BaseModel):
    max_instructions: int = Field(default=100)
    instruction_generator_parameters: InstructionGeneratorParameters = Field(
        default_factory=lambda: InstructionGeneratorParameters()
    )


class QLearningConsts(BaseModel):
    alpha: float = Field(default=0.1)
    gamma: float = Field(default=0.9)
    epsilon: float = Field(default=0.1)
    alpha_decay: float = Field(default=0.99)
    epsilon_decay: float = Field(default=0.99)


class QLearningProgramParameters(BaseModel):
    program_parameters: ProgramParameters = Field(default_factory=lambda: ProgramParameters())
    consts: QLearningConsts = Field(default_factory=lambda: QLearningConsts())


class LGPConfig(BaseModel):
    default_fitness: float = Field(default=0.0)
    population_size: int = Field(default=100)
    gap: float = Field(default=0.5)
    mutation_percent: float = Field(default=0.1)
    crossover_percent: float = Field(default=0.1)
    n_generations: int = Field(default=200)
    n_trials: int = Field(default=1)
    seed: int | None = Field(default=None)
    program_parameters: ProgramParameters = Field(default_factory=lambda: ProgramParameters())


class QLearningConfig(BaseModel):
    default_fitness: float = Field(default=0.0)
    population_size: int = Field(default=100)
    gap: float = Field(default=0.5)
    mutation_percent: float = Field(default=0.1)
    crossover_percent: float = Field(default=0.1)
    n_generations: int = Field(default=200)
    n_trials: int = Field(default=1)
    seed: int | None = Field(default=None)
    program_parameters: QLearningProgramParameters = Field(
        default_factory=lambda: QLearningProgramParameters()
    )


# =============================================================================
# Program Output Models (for parsing JSON output from Rust)
# =============================================================================


class InstructionMode(str, Enum):
    """Mode of instruction - external uses input values, internal uses registers."""

    EXTERNAL = "External"
    INTERNAL = "Internal"


class InstructionOp(str, Enum):
    """Arithmetic operation for an instruction."""

    ADD = "Add"
    SUB = "Sub"
    MULT = "Mult"
    DIVIDE = "Divide"


class Instruction(BaseModel):
    """A single instruction in a program."""

    src_idx: int
    tgt_idx: int
    mode: InstructionMode
    op: InstructionOp
    external_factor: float


class Registers(BaseModel):
    """Register state of a program."""

    data: list[float | None]
    n_actions: int


class Program(BaseModel):
    """A complete program with instructions and registers."""

    id: str = Field(default="")
    instructions: list[Instruction] = Field(default_factory=list)
    registers: Registers | None = None
    fitness: float = Field(default=0.0)

    # For Q-learning wrapper format (nested program)
    program: Program | None = None

    @classmethod
    def from_json(cls, path: Path) -> Program:
        """Load program from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def get_effective_fitness(self) -> float:
        """Get fitness, handling Q-learning wrapper format."""
        if self.program is not None:
            return self.program.fitness
        return self.fitness


class Population(BaseModel):
    """A population is a list of generations, each containing programs."""

    generations: list[list[Program]]

    @classmethod
    def from_json(cls, path: Path) -> Population:
        """Load population from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(generations=[[Program.model_validate(p) for p in gen] for gen in data])

    def get_fitness_by_generation(self) -> list[list[float]]:
        """Extract fitness scores grouped by generation."""
        return [[p.get_effective_fitness() for p in gen] for gen in self.generations]


class ExperimentOutput(BaseModel):
    """Complete output from an experiment run."""

    best: Program
    median: Program
    worst: Program
    params: LGPConfig | QLearningConfig
    population: Population

    @classmethod
    def from_directory(cls, path: Path) -> ExperimentOutput:
        """Load all experiment outputs from a directory."""
        best = Program.from_json(path / "best.json")
        median = Program.from_json(path / "median.json")
        worst = Program.from_json(path / "worst.json")
        population = Population.from_json(path / "population.json")

        with open(path / "params.json") as f:
            params_data = json.load(f)

        # Detect Q-learning vs LGP based on nested structure
        if "program_parameters" in params_data.get("program_parameters", {}):
            params = QLearningConfig.model_validate(params_data)
        else:
            params = LGPConfig.model_validate(params_data)

        return cls(
            best=best,
            median=median,
            worst=worst,
            params=params,
            population=population,
        )
