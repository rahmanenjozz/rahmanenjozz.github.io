# rahmanenjozz.github.io

My personal site and blog, live at [rahmanenjozz.github.io](https://rahmanenjozz.github.io).

Built with Jekyll, hosted on GitHub Pages. Posts are written in Bahasa Indonesia and English.

## Structure

```
_config.yml          site settings, social links, per-language defaults
_data/i18n.yml       small UI strings per language
_includes/           head (SEO), post row, translation lookup, feed template
_layouts/            default, post
_posts/id/           Indonesian posts  -> /id/<slug>/
_posts/en/           English posts     -> /en/<slug>/
assets/css/main.css
assets/img/posts/<slug>/
index.html           home
blog.html            /blog/ with language filter
about.html           /about/
feed-id.xml, feed-en.xml
```

## Writing a post

Create `_posts/id/YYYY-MM-DD-slug.md` (or `_posts/en/`):

```yaml
---
title: Judul tulisan
description: Satu kalimat untuk meta description dan daftar blog.
ref: slug            # same value in both languages links the translations
tags: [nlp]
image: /assets/img/posts/slug/cover.jpg   # optional, for social preview
medium: https://medium.com/...            # optional, after importing
---
```

`lang` and the URL are set automatically from the folder. A translation is optional: if a post with the same `ref` exists in the other language, the ID/EN switch and `hreflang` tags appear.

## Cross-posting to Medium

Publish here first, then use Medium's *Import story* with the post URL (one import per language). Medium sets the canonical link back to this site. Add the Medium URL to the `medium` field afterwards.

## Local preview

```
bundle install
bundle exec jekyll serve
```
