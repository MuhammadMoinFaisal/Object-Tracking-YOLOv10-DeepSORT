---
comments: true
description: Learn about Ultralytics Explorer GUI for semantic search, SQL queries, and AI-powered natural language search in CV datasets.
keywords: Ultralytics, Explorer GUI, semantic search, vector similarity search, AI queries, SQL queries, computer vision, dataset exploration, image search, OpenAI integration
---

# Explorer GUI

Explorer GUI is like a playground build using [Ultralytics Explorer API](api.md). It allows you to run semantic/vector similarity search, SQL queries and even search using natural language using our ask AI feature powered by LLMs.

<p>
    <img width="1709" alt="Explorer Dashboard Screenshot 1" src="https://github.com/ultralytics/ultralytics/assets/15766192/feb1fe05-58c5-4173-a9ff-e611e3bba3d0">
</p>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo?start=306"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Explorer Dashboard Overview
</p>

### Installation

```bash
pip install ultralytics[explorer]
```

!!! note "Note"

    Ask AI feature works using OpenAI, so you'll be prompted to set the api key for OpenAI when you first run the GUI.
    You can set it like this - `yolo settings openai_api_key="..."`

## Semantic Search / Vector Similarity Search

Semantic search is a technique for finding similar images to a given image. It is based on the idea that similar images will have similar embeddings. In the UI, you can select one of more images and search for the images similar to them. This can be useful when you want to find images similar to a given image or a set of images that don't perform as expected.

For example:
In this VOC Exploration dashboard, user selects a couple airplane images like this:
<p>
<img width="1710" alt="Explorer Dashboard Screenshot 2" src="https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/3becdc1d-45dc-43b7-88ff-84ff0b443894">
</p>

On performing similarity search, you should see a similar result:
<p>
<img width="1710" alt="Explorer Dashboard Screenshot 3" src="https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/aeea2e16-bc2b-41bb-9aef-4a33bfa1a800">
</p>

## Ask AI

This allows you to write how you want to filter your dataset using natural language. You don't have to be proficient in writing SQL queries. Our AI powered query generator will automatically do that under the hood. For example - you can say - "show me 100 images with exactly one person and 2 dogs. There can be other objects too" and it'll internally generate the query and show you those results. Here's an example output when asked to "Show 10 images with exactly 5 persons" and you'll see a result like this:
<p>
<img width="1709" alt="Explorer Dashboard Screenshot 4" src="https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/55a67181-3b25-4d2f-b786-2a6a08a0cb6b">
</p>

Note: This works using LLMs under the hood so the results are probabilistic and might get things wrong sometimes

## Run SQL queries on your CV datasets

You can run SQL queries on your dataset to filter it. It also works if you only provide the WHERE clause. Example SQL query would show only the images that have at least one 1 person and 1 dog in them:

```sql
WHERE labels LIKE '%person%' AND labels LIKE '%dog%'
```

<p>
<img width="1707" alt="Explorer Dashboard Screenshot 5" src="https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/14fbb237-0b2d-4b7c-8f62-2fca4e6cc26f">
</p>

This is a Demo build using the Explorer API. You can use the API to build your own exploratory notebooks or scripts to get insights into your datasets. Learn more about the Explorer API [here](api.md).
