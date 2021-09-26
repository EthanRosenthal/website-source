from collections import Counter, defaultdict
from typing import Dict, List, Tuple


from sandmat.models import Ingredient, Sandwich


def get_ranked_categories(
    sandwiches: List[Sandwich],
) -> Dict[str, Dict[Ingredient, int]]:
    """
    Group ingredients by category and order them by how many sandwiches
    the ingredient is in.

    For ranked_categories, the key is the category name (e.g. 'meat'),
    and the value is an ordered dict where key is Ingredient and value
    is the number of the sandwiches this ingredient is in (i.e. the "ingredient rank").
    The dict is in descending order of ingredient rank.
    """
    category_order = ["meat", "cheese", "topping", "dressing"]

    category_counts = defaultdict(Counter)
    for sandwich in sandwiches:
        for ingredient in sandwich.ingredients:
            category_counts[ingredient.category].update([ingredient])

    # Dicts in Python 3.7+ are ordered based on insertion order.
    # Thus, iterating through ranked_categories will iterate in
    # the order of category_order
    ranked_categories = {}
    for category in category_order:
        ranked_categories[category] = {
            ingredient: rank
            for rank, (ingredient, count) in enumerate(
                category_counts[category].most_common(), start=1
            )
        }
    return ranked_categories


def _sort_key(
    sandwich: Sandwich, ranked_categories: Dict[str, Ingredient]
) -> Tuple[int, ...]:
    """
    Given a sandwich, return a key to sort this sandwich with respect to.
    The key is a tuple of ingredient ranks where there is a different
    element of the tuple for each category. This allows for sorting on
    multiple fields i.e. sort by meat, then cheese, then topping,
    then dressing.
    """
    sortings = []
    for category, ranked_ingredients in ranked_categories.items():
        ranks = [
            ranked_ingredients[i]
            for i in sandwich.ingredients
            if i in ranked_ingredients
        ]
        if ranks:
            category_rank = min(ranks)
        else:
            # No sandwich ingredients from this category.
            # Set rank to max rank + 1 for the category
            category_rank = len(ranked_ingredients) + 1
        sortings.append(category_rank)
    return tuple(sortings)


def get_ordered_sandwiches(
    sandwiches: List[Sandwich], ranked_categories: Dict[str, Dict[Ingredient, int]]
) -> List[Sandwich]:
    return sorted(
        sandwiches, key=lambda sandwich: _sort_key(sandwich, ranked_categories)
    )


def get_ordered_ingredients(
    ranked_categories: Dict[str, Dict[Ingredient, int]]
) -> Dict[str, Ingredient]:
    """
    We rely on the fact that dicts are ordered by insertion order,
    so iterating through ordered_ingredients will iterate through
    the ingredients by category order and then ingredient rank.
    """
    ordered_ingredients: Dict[str, Ingredient] = {}
    for category, ingredients in ranked_categories.items():
        for ingredient in ingredients:
            ordered_ingredients[ingredient.name] = ingredient
    return ordered_ingredients
