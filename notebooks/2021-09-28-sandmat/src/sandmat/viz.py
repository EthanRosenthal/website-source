from typing import Dict, List, Tuple

import altair as alt
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper


from sandmat.models import Ingredient, Sandwich


def make_sandwich_matrix(
    ordered_sandwiches: List[Sandwich], ordered_ingredients: Dict[str, Ingredient]
) -> np.ndarray:
    """
    Make matrix where rows are sandwiches and columns are ingredients.
    Values are 1 if sandwich has the ingredient and 0 otherwise.
    """
    sandwich_mat = np.zeros((len(ordered_sandwiches), len(ordered_ingredients)))
    sandwich_to_idx = {sandwich: idx for idx, sandwich in enumerate(ordered_sandwiches)}
    ingredient_to_idx = {
        ingredient: idx for idx, ingredient in enumerate(ordered_ingredients.values())
    }
    for sandwich in ordered_sandwiches:
        for ingredient in sandwich.ingredients:
            sandwich_mat[sandwich_to_idx[sandwich], ingredient_to_idx[ingredient]] = 1
    return sandwich_mat


def column_colors(ordered_ingredients: dict) -> np.ndarray:
    color_mapper = {
        "meat": "red",
        "cheese": "orange",
        "topping": "green",
        "dressing": "purple",
    }
    return np.array([color_mapper[i.category] for i in ordered_ingredients.values()])


def plot_sandwiches(
    sandwich_mat: np.ndarray,
    ordered_sandwiches: List[Sandwich],
    ordered_ingredients: Dict[str, Ingredient],
) -> Tuple[Figure, Axes]:

    fig, ax = plt.subplots(figsize=(10, 20))
    xs, ys = np.where(sandwich_mat)
    colors = column_colors(ordered_ingredients)

    ax.scatter(x=ys, y=xs, marker="x", c=colors[ys])
    ax.set_yticks(np.arange(len(sandwich_mat)))
    ax.set_yticklabels([s.name for s in ordered_sandwiches], fontsize=12)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(sandwich_mat.shape[1]))
    ax.set_xticklabels(
        list(ordered_ingredients.keys()), rotation=45, ha="left", fontsize=12
    )
    ax.xaxis.tick_top()

    for i in range(-1, len(sandwich_mat)):
        ax.plot(
            (-0.5, sandwich_mat.shape[1] + 0.5),
            (0.5 + i, 0.5 + i),
            c="gray",
            alpha=0.5,
            linestyle="dashed",
            linewidth=0.5,
        )

    ax.set_xlim((-0.5, sandwich_mat.shape[1]))

    for i in range(-1, len(sandwich_mat)):
        ax.plot(
            (0.5 + i, 0.5 + i),
            (-0.5, sandwich_mat.shape[0] + 0.5),
            c="gray",
            alpha=0.5,
            linestyle="dashed",
            linewidth=0.5,
        )
    ax.set_ylim(-0.5, len(sandwich_mat) - 0.5)
    # ax.grid(b=True, which="both");
    ax.invert_yaxis()

    for color, tick in zip(colors, ax.get_xticklabels()):
        tick.set_color(color)
    return fig, ax


def plot_ingredients(
    ordered_ingredients: Dict[str, Ingredient], model_result: RegressionResultsWrapper
) -> alt.Chart:
    data = {
        "values": [
            {
                "x": x,
                "Ingredient": ingredient.name,
                "Price": price,
                "Category": ingredient.category,
                "SE": se,
            }
            for x, (ingredient, price, se) in enumerate(
                zip(
                    list(ordered_ingredients.values()),
                    model_result.params[1:],
                    model_result.bse[1:],
                )
            )
        ]
    }
    color_mapper = {
        "meat": "red",
        "cheese": "orange",
        "topping": "green",
        "dressing": "purple",
    }
    scale = alt.Scale(
        domain=list(color_mapper.keys()), range=list(color_mapper.values())
    )
    chart = (
        alt.Chart(data)
        .mark_bar(opacity=0.5)
        .encode(
            x=alt.X("Price:Q", axis=alt.Axis(format="$.2f")),
            y=alt.Y("Ingredient:N", sort="-x"),
            color=alt.Color("Category:N", scale=scale),
            tooltip=[
                "Ingredient:N",
                alt.Tooltip("Price:Q", format="$.2f"),
                alt.Tooltip("SE:Q", format="$.2f", title="Standard Error"),
            ],
        )
    )
    return chart


def plot_actual_vs_pred(
    y_actual: np.ndarray, y_pred: np.ndarray, ordered_sandwiches: Dict[str, Sandwich]
) -> alt.LayerChart:
    data = {
        "values": [
            {
                "Price": price,
                "Prediction": prediction,
                "Sandwich": sandwich.name,
                "Ingredients": ", ".join(
                    ingredient.name for ingredient in sandwich.ingredients
                ),
                "Vegetarian": all(
                    ingredient.category != "meat" for ingredient in sandwich.ingredients
                ),
            }
            for (price, prediction, sandwich) in zip(
                y_actual, y_pred, ordered_sandwiches
            )
        ]
    }

    domain = y_actual.min() - 0.5, y_actual.max() + 0.5
    sandwich_chart = (
        alt.Chart(data)
        .mark_point()
        .encode(
            x=alt.X(
                "Price:Q",
                scale=alt.Scale(domain=domain),
                axis=alt.Axis(format="$.2f"),
            ),
            y=alt.Y(
                "Prediction:Q",
                scale=alt.Scale(domain=domain),
                axis=alt.Axis(format="$.2f"),
            ),
            color="Vegetarian:N",
            tooltip=["Sandwich:N", "Ingredients:N"],
        )
    )

    one_to_one = {
        "values": [
            {"x": x_, "y": y_}
            for x_, y_ in zip(np.linspace(*domain, 101), np.linspace(*domain, 101))
        ]
    }
    line_chart = (
        alt.Chart(one_to_one)
        .mark_line(color="black", strokeDash=[5, 5])
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title="Price")),
            y=alt.Y("y:Q", axis=alt.Axis(title="Prediction")),
        )
    )
    return sandwich_chart + line_chart
