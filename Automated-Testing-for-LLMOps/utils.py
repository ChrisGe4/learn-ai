import github
import os
import requests
import random
from dotenv import load_dotenv, find_dotenv
from yaml import safe_dump, safe_load

adjectives = [
    "adoring",
    "affirmative",
    "appreciated",
    "available",
    "best-selling",
    "blithe",
    "brightest",
    "charismatic",
    "convincing",
    "dignified",
    "ecstatic",
    "effective",
    "engaging",
    "enterprising",
    "ethical",
    "fast-growing",
    "glad",
    "hardy",
    "idolized",
    "improving",
    "jubilant",
    "knowledgeable",
    "long-lasting",
    "lucky",
    "marvelous",
    "merciful",
    "mesmerizing",
    "problem-free",
    "resplendent",
    "restored",
    "roomier",
    "serene",
    "sharper",
    "skilled",
    "smiling",
    "smoother",
    "snappy",
    "soulful",
    "staunch",
    "striking",
    "strongest",
    "subsidized",
    "supported",
    "supporting",
    "sweeping",
    "terrific",
    "unaffected",
    "unbiased",
    "unforgettable",
    "unrivaled",
]

nouns = [
    "agustinia",
    "apogee",
    "bangle",
    "cake",
    "cheese",
    "clavicle",
    "client",
    "clove",
    "curler",
    "draw",
    "duke",
    "earl",
    "eustoma",
    "fireplace",
    "gem",
    "glove",
    "goal",
    "ground",
    "jasmine",
    "jodhpur",
    "laugh",
    "message",
    "mile",
    "mockingbird",
    "motor",
    "phalange",
    "pillow",
    "pizza",
    "pond",
    "potential",
    "ptarmigan",
    "puck",
    "puzzle",
    "quartz",
    "radar",
    "raver",
    "saguaro",
    "salary",
    "sale",
    "scarer",
    "skunk",
    "spatula",
    "spectacles",
    "statistic",
    "sturgeon",
    "tea",
    "teacher",
    "wallet",
    "waterfall",
    "wrinkle",
]


def inspect_config():
  with open("circle_config.yml") as f:
    print(safe_dump(safe_load(f)))


def load_env():
  _ = load_dotenv(find_dotenv())


def get_google_api_key():
  load_env()
  return os.getenv("GOOGLE_API_KEY")


def get_circle_api_key():
  load_env()
  circle_token = os.getenv("CIRCLE_TOKEN")
  return circle_token


def get_gh_api_key():
  load_env()
  github_token = os.getenv("GH_TOKEN")
  return github_token


def get_repo_name():
  return "ChrisGe4/automated-testing-for-llmops"


def get_branch() -> str:
  prefix = "dl-cci"
  adjective = random.choice(adjectives)
  noun = random.choice(nouns)
  number = random.choice(range(1, 100))

  return f"{prefix}-{adjective}-{noun}-{number}"
