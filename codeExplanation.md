[Table of contents](tableOfContents.md)

You can find the Github page for this project [here](https://github.com/ZermbaGerd/gavin194final.github.io). This page includes the documents that make up this website, as well as the code to run your own version of the LLM.

Below is a guide to navigating the codebase, as well as a guide for setting up the code.

## Requirements
In order to run the python code, you'll need to install these packages:

- !pip install langchain
- !pip install sentence-transformers
- !pip install faiss-cpu
- !pip install bs4
- !pip install replicate
- !pip install langchain-community

## References
The python code is largely from [this tutorial](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/RAG/hello_llama_cloud.ipynb).

## Setting up the LLM
That tutorial runs a larger model on a server in the cloud, but you don't have to do that. Those servers can be very useful for running larger models that require much stronger computers than the typical consumer would have, but they have limited access, and you start needing to pay after a certain point.

In my case, I am running a local copy of Meta's Llama 3.1 (8b), instead of a cloud-compute model hosted by a service like Groq or Replicate.

The tutorial for how to run a local copy of Llama can be found [here](https://medium.com/@paulocsb/running-llama-3-1-locally-with-ollama-a-step-by-step-guide-44c2bb6c1294).

However you end up accessing your LLM - through cloud-compute or a local instance, you can just put the API call for it into the LLM section and the other parts of the code will still work.

## Picking your supporting documents

In the example code, I've included a few articles that the AI will draw its information from. Those articles are also listed on the website [here](listOfSources.md). You can add your own to this list, but it will increase the time it takes to run the model. Also, make sure that the source you're using has an open-access license, or you might be reflecting the same problems I discuss in my [reflections](reflections.md)!

## Picking your prompt details

Most LLM's that are trying to serve a particular purpose include extra boilerplate in the form of system prompts, which are overarching instructions that the system always gives to the model alongside user prompts. This code does the same, but with a prefix to every user prompt. Feel free to play around with that prefix to see if you can change the behavior of the bot.