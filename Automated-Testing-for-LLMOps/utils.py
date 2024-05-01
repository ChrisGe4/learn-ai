import time

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


def create_tree_element(repo, path, content):
  blob = repo.create_git_blob(content, "utf-8")
  element = github.InputGitTreeElement(path=path, mode="100644", type="blob",
                                       sha=blob.sha)

  return element


def push_files(repo_name, branch_name, files):
  files_to_push = set(files)
  # include the config.yaml file
  g = github.Github(os.environ["GH_TOKEN"])
  repo = g.get_repo(repo_name)

  elements = []
  config_element = create_tree_element(repo, ".circleci/config.yml",
                                       open("circle_config.yaml").read())
  elements.append(config_element)

  requirements_element = create_tree_element(repo, "requirements.txt",
                                             open("requirements.txt"))

  elements.append(requirements_element)

  for file in files_to_push:
    print(f"uploading {file}")
    with open(file, encoding="utf-8") as f:
      content = f.read()
      element = create_tree_element(repo, file, content)
      elements.append(element)

  head_sha = repo.get_branch("main").commit.sha

  print(f"pushing files to: {branch_name}")
  try:
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=head_sha)
    time.sleep(2)
  except Exception as _:
    print(
      f"{branch_name} already exists in the repository pushing updated changes")

  branch_sha = repo.get_branch(branch_name).commit.sha

  base_tree = repo.get_git_tree(sha=branch_sha)
  tree = repo.create_git_tree(elements, base_tree)
  parent = repo.get_git_commit(sha=branch_sha)
  commit = repo.create_git_commit("Trigger CI evaluation pipeline", tree,
                                  [parent])
  branch_refs = repo.get_git_ref(f"heads/{branch_name}")
  branch_refs.edit(sha=commit.sha)

def _trigger_circle_pipeline(repo_name,branch, token, params=None):
  params = {} if params is None else params
  r = requests.post(
      f"{os.getenv('DLAI_CIRCLE_CI_API_BASE', 'https://circleci.com')}/api/v2/project/gh/{repo_name}/pipeline",
      headers={"Circle-Token": f"{token}", "accept": "application/json"},
      json={"branch": branch, "parameters": params},
  )
  pipeline_data = r.json()
  pipeline_number = pipeline_data["number"]
  print(
      f"Please visit https://app.circleci.com/pipelines/github/{repo_name}/{pipeline_number}"
  )

def trigger_commit_evals(repo_name, branch, token):
  _trigger_circle_pipeline(repo_name, branch, token, {"eval-mode": "commit"})


def trigger_release_evals(repo_name, branch, token):
  _trigger_circle_pipeline(repo_name, branch, token, {"eval-mode": "release"})

def trigger_full_evals(repo_name, branch, token):
  _trigger_circle_pipeline(repo_name, branch, token, {"eval-mode": "full"})