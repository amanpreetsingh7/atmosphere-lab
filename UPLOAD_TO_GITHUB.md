# Upload Atmosphere Lab to its own repository

1. Extract the archive and keep the folder named `atmosphere-lab`. Use this clean folder as the repository root. Your existing research installation and the focused contrast repository are separate projects.
2. In GitHub Desktop, choose **File → Add Local Repository** and select the extracted folder. If Desktop reports that it is not yet a repository, choose the option to create a repository there. Confirm the path points to the extracted folder itself, where `README.md` and `app.py` are located.
3. Review the files in **Changes** and commit them with a message such as “Add standalone Atmosphere Lab app.” Git includes the hidden configuration directories automatically.
4. Choose **Publish repository** and use the name `atmosphere-lab`. Select private or public visibility according to whom you want to have access.
5. On GitHub, confirm `README.md`, `app.py`, `atmosphere_compare/`, `configs/` and `tests/` are at the repository root. The **Actions** tab will show the software checks once they run.

The supplied source folder contains no model downloads. Keep your models, personal catalog entries, imported response files and generated results on your laptop. The starter `configs/app_catalog.json` is a tracked template; if the app updates it locally, leave those changes out of your commits. Do not upload your entire existing research folder or an old application archive.

If using GitHub's web upload instead, drag the extracted contents to the repository root. If the uploader rejects hidden folders, create `.streamlit/config.toml` and `.github/workflows/tests.yml` individually using **Add file → Create new file**, then paste the contents of the corresponding local files.
