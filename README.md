# Eternia Vision AI

## Run with Docker

Build the image from the project root:

```sh
docker build -t eternia-vision-ai .
```

Start the API on port 5000:

```sh
docker run --rm -p 5000:5000 eternia-vision-ai
```

The API is available at `http://localhost:5000`; interactive API documentation
is at `http://localhost:5000/docs`.
