from typing import Iterable, List, Tuple
from urllib.request import urlopen

from bs4 import BeautifulSoup, Tag

from sandmat.models import Ingredient, Sandwich


def get_sandwich_menu(soup: BeautifulSoup) -> Tag:
    sections = soup.find_all(name="section", **{"class": "menu-section"})
    menu = None
    for section in sections:
        if section.find("h2").text == "Menu":
            menu = section

    if menu is None:
        raise ValueError("Cannot find Sandwich section of menu.")
    return menu


def get_sandwich_tags(menu: Tag) -> Iterable[Tag]:
    return menu.find_all(name="li", **{"class": "menu-item"})


def get_sandwich_name(sandwich_tag: Tag) -> str:
    return sandwich_tag.find(name="p", **{"class": "menu-item__heading"}).text


def get_sandwich_price(sandwich_tag: Tag) -> float:
    price = sandwich_tag.find(name="p", **{"class": "menu-item__details--price"}).text
    price = float(price.strip(" $\n"))
    return price


def cleanup_ingredient(name: str) -> str:
    name = name.lower()
    mapper = {
        "m. bel paese cheese": "m. bel paese",
        "arugul": "arugula",
        "artichoke": "artichokes",
        "sweet roasted pepper": "sweet roasted peppers",
        "olive past": "olive paste",
    }
    return mapper.get(name, name)


def categorize_ingredient(name: str) -> str:
    mapper = {
        "artichokes": "topping",
        "arugula": "topping",
        "capicollo": "meat",
        "caponata of eggplant": "topping",
        "dressing": "dressing",
        "fresh mozzarella": "cheese",
        "hot peppers": "topping",
        "m. bel paese": "cheese",
        "mortadella": "meat",
        "olive paste": "dressing",
        "prosciutto": "meat",
        "provolone cheese": "cheese",
        "salami": "meat",
        "sardines or mackerel": "meat",
        "smoked chicken breast": "meat",
        "smoked mozzarella": "cheese",
        "sopressata": "meat",
        "sun dried tomatoes": "topping",
        "sweet roasted peppers": "topping",
        "tuna": "meat",
    }
    try:
        return mapper[name]
    except KeyError:
        raise ValueError(f"Category not found for ingredient {name!r}")


def get_sandwich_ingredients(sandwich_tag: Tag) -> Tuple[Ingredient]:
    other_paragraphs = sandwich_tag.find_all(name="p")
    ingredients = None
    for p in other_paragraphs:
        if not p.get("class"):
            ingredients = p.text.split(", ")
            ingredients = [cleanup_ingredient(ingredient) for ingredient in ingredients]
            ingredients = tuple(
                [
                    Ingredient(
                        name=ingredient, category=categorize_ingredient(ingredient)
                    )
                    for ingredient in ingredients
                ]
            )
    if ingredients is None:
        raise ValueError(f"Cannot parse ingredients for sandwich {sandwich_tag}")
    return ingredients


def convert_sandwich_tag_to_sandwich(sandwich_tag: Tag) -> Sandwich:
    name = get_sandwich_name(sandwich_tag)
    ingredients = get_sandwich_ingredients(sandwich_tag)
    price = get_sandwich_price(sandwich_tag)
    return Sandwich(name, ingredients, price)


def get_sandwiches(url: str) -> List[Sandwich]:
    soup = BeautifulSoup(urlopen(url), features="html.parser")
    menu = get_sandwich_menu(soup)
    sandwich_tags = get_sandwich_tags(menu)
    sandwiches = [
        convert_sandwich_tag_to_sandwich(sandwich_tag) for sandwich_tag in sandwich_tags
    ]
    return sandwiches
