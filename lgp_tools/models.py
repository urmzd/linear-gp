"""Pydantic models for parsing Rust output."""

from pydantic import BaseModel


class InstructionGeneratorParameters(BaseModel):
    n_extras: int
    external_factor: float
    n_actions: int
    n_inputs: int


class ProgramParameters(BaseModel):
    max_instructions: int
    instruction_generator_parameters: InstructionGeneratorParameters


class QLearningConsts(BaseModel):
    alpha: float
    gamma: float
    epsilon: float
    alpha_decay: float
    epsilon_decay: float


class QLearningProgramParameters(BaseModel):
    program_parameters: ProgramParameters
    consts: QLearningConsts


class LGPConfig(BaseModel):
    default_fitness: float
    population_size: int
    gap: float
    mutation_percent: float
    crossover_percent: float
    n_generations: int
    n_trials: int
    seed: int | None
    program_parameters: ProgramParameters


class QLearningConfig(BaseModel):
    default_fitness: float
    population_size: int
    gap: float
    mutation_percent: float
    crossover_percent: float
    n_generations: int
    n_trials: int
    seed: int | None
    program_parameters: QLearningProgramParameters


class Program(BaseModel):
    fitness: float


class PopulationGeneration(BaseModel):
    programs: list[Program]
