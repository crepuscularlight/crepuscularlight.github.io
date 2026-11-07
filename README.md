# Liudi Yang’s homepage and blog

## Publish on GitHub Pages

1. Sign in as crepuscularlight. Create a public repository named `crepuscularlight.github.io` with a README and a `main` branch, or open that repository if you already have it. If it already contains a site, merge the files you want on a branch and review them before merging into main.
2. Unzip this starter. Upload the contents with **Add file → Upload files** in the repository. Drag the folders along with the files so `_layouts`, `_posts`, `assets`, and `blog` keep their structure. The repository root should contain `_config.yml` and `index.html` directly. Review the sample blog post before publishing.
3. Commit to `main`. Choose **Settings → Pages → Build and deployment → Deploy from a branch**, select **main** and **/(root)**, and save. The source branch must have Jekyll enabled (remove a legacy `.nojekyll` file if present).
4. After the Pages deployment succeeds, open https://crepuscularlight.github.io/ and follow **Blog**. The separate RoboEnvision project is linked at its existing URL.

## Make it yours

Edit `index.html` to update the biography and selected papers. The portrait currently loads from your Freiburg lab profile. To host your own photo, upload it to `assets` and change the image source in `index.html`. Edit `_config.yml` for the display name and profile links. The shared styling lives in `assets/site.css`.

## Add a blog post

In the GitHub repository choose **Add file → Create new file** and enter `_posts/YYYY-MM-DD-short-title.md`, using the actual publication date. Start the file with:

```markdown
---
layout: post
title: "A title for my next research note"
description: "One sentence for the blog index."
---

Write the post here in Markdown.
```

Commit the file. Jekyll creates a page for it, lists it on `/blog/` newest first, and shows the newest post on the homepage. The included post is an example draft for you to edit, replace or delete before uploading. Use a present or past date for immediate publication.

## Sources

- [University of Freiburg lab profile](https://rl.informatik.uni-freiburg.de/people/yang/)
- [RoboEnvision](https://crepuscularlight.github.io/roboenvision/)
- [LiDAR loop closure paper](https://doi.org/10.1007/s10846-025-02223-6) and [repository](https://github.com/crepuscularlight/SemanticLoopClosure)
- [GitHub Pages setup](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)
- [Jekyll posts](https://jekyllrb.com/docs/posts/)
