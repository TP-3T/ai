# ai
Repository for the AI features

## Contributing and Branches:

  - There are four types of branches:
    - feature branches
    - ai
    - dev
    - main

  - The feature branches are for developing seperate features. They should be branched off from the `ai` branch, and then merged back into `ai`
    - Feature branches should follow the format: `ai/<feature-or-scope>`
    - Example feature branch name: `ai/add-climate-prediction-data`

  - The branch `ai` is for merging our ai features. It requires a pull request to merge feature branches into it, but no code reviewers

  - The branch `dev` is for integrating our team's components with the other teams'. It requires a pull request to merge, as well as a code review by 2 people (1 from group of reviewers, and 1 from a person who didn't make the pull request)

  - The branch `main` is for merging working and tested code from dev. It requires the same as dev (pull request and 2 code reviewers)
