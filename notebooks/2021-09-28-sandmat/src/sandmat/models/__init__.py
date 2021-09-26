from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Ingredient:
    name: str
    category: str


@dataclass(frozen=True)
class Sandwich:
    name: str
    ingredients: Tuple[Ingredient]
    price: float
