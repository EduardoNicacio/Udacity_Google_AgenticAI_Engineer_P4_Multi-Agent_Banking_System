import os
from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage

# Load environment variables from .env
load_dotenv()


def fetch_gcs_content(gcs_uri: str) -> str:
    """
    Fetch the text content of a file from Google Cloud Storage.

    Args:
        gcs_uri (str): A GCS URI in the form gs://bucket-name/object-name

    Returns:
        str: The text content of the file, or an empty string on failure
    """
    try:
        # Strip the gs:// prefix and split into bucket and blob path
        without_prefix = gcs_uri.replace("gs://", "")
        bucket_name, _, blob_name = without_prefix.partition("/")

        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text()
    except Exception as e:
        return f"[Could not fetch content from {gcs_uri}: {e}]"


# Definition of a tool that accesses a Vertex AI Search Datastore
#
# This is based on code provided by Google at
# https://cloud.google.com/generative-ai-app-builder/docs/samples/genappbuilder-search
#
# The object definitions aren't available to all IDEs because of Google's ProtoBuf
# implementation, so the IDE may generate a warning, but work fine. I've used
# dicts here instead, but indicated the Class that could be used instead.
# You can see the definitions at
# https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types
#
def search(
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str,
) -> list[str]:
    """
    Search Vertex AI Search Datastore for relevant documents and content.

    This function performs a search query against the Vertex AI Search Datastore,
    retrieving document metadata and optionally fetching raw content from Google Cloud Storage (GCS)
    when available. It includes deduplication logic to avoid duplicate results from different file formats
    of the same document.

    Args:
        project_id (str): The Google Cloud project ID containing the search engine.
        location (str): The geographic region where the search engine is located.
                        Common values include "global" or specific regions like "us-central1".
        engine_id (str): The unique identifier for the search engine within the project.
        search_query (str): The text query to search for in the datastore.

    Returns:
        list[str]: A list of strings containing search results in the format:
                   - For documents with fetchable content: "Title: {title}\nSource: {link}\nContent:\n{content}"
                   - For documents without fetchable content or non-text files: "Title: {title}\nSource: {link}"
                   Returns ["No results found."] if no matching documents are found.

    Raises:
        Exception: If the search client cannot be created, request fails, or any other error occurs during execution.

    Example:
        >>> results = search(
        ...     project_id="my-project",
        ...     location="us-central1",
        ...     engine_id="bird-boutique-search-engine",
        ...     search_query="parrot care guide"
        ... )
        >>> for result in results:
        ...     print(result)

    Notes:
        - Automatically deduplicates documents that appear as both .txt and .pdf versions.
        - Only fetches GCS content when the datastore flags it as fetchable AND the link points to a plain-text file (.txt).
        - Falls back to metadata-only results for non-text files or non-fetchable documents.
        - Includes spell correction and query expansion enabled by default.
    """

    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    client = discoveryengine.SearchServiceClient(client_options=client_options)

    serving_config = (
        f"projects/{project_id}/locations/{location}/collections/default_collection"
        f"/engines/{engine_id}/servingConfigs/default_config"
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=5,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO,
        ),
    )

    page_result = client.search(request)

    results = []
    seen_content = set()  # deduplicate .txt and .pdf versions of the same document

    for result in page_result:
        struct_data = result.document.derived_struct_data

        title_str = struct_data.get("title", "Unknown title")  # type: ignore
        link_str = struct_data.get("link", "Unknown source")  # type: ignore
        can_fetch = struct_data.get("can_fetch_raw_content", "false")  # type: ignore

        # Only fetch GCS content when the datastore flagged it as fetchable
        # and the link points to a plain-text file to avoid binary content
        if can_fetch == "true" and link_str.endswith(".txt"):
            content = fetch_gcs_content(link_str)

            # Deduplicate: same document may appear as both .txt and .pdf
            if content and content not in seen_content:
                seen_content.add(content)
                results.append(
                    f"Title: {title_str}\nSource: {link_str}\nContent:\n{content}"
                )
        else:
            # Fall back to metadata only for non-text files
            results.append(f"Title: {title_str}\nSource: {link_str}")

    return results if results else ["No results found."]


def datastore_search_tool(search_query: str) -> list[str]:
    """
    Search the Vertex AI Search Datastore for information about the National Bank.

    Args:
        search_query (str): The search query to look up in documents

    Returns:
        list[str]: List of document snippets relevant to the query
    """
    project_id = os.getenv("DATASTORE_PROJECT_ID", "")
    engine_id = os.getenv("DATASTORE_ENGINE_ID", "")
    location = os.getenv("DATASTORE_LOCATION", "global")

    try:
        return search(project_id, location, engine_id, search_query)
    except Exception as e:
        return [f"A problem occurred: {e}"]
