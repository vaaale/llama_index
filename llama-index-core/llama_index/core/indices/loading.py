import logging
from typing import Any, List, Optional, Sequence, Dict

from llama_index.core.data_structs import IndexList, IndexDict, KeywordTable, IndexGraph
from llama_index.core.data_structs.data_structs import KG
from llama_index.core.indices.base import BaseIndex, IS
from llama_index.core.indices.composability.graph import ComposableGraph
from llama_index.core.indices.registry import INDEX_STRUCT_TYPE_TO_INDEX_CLASS
from llama_index.core.schema import IndexNode
from llama_index.core.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)


def load_index_from_storage(
        storage_context: StorageContext,
        index_id: Optional[str] = None,
        **kwargs: Any,
) -> BaseIndex:
    """Load index from storage context.

    Args:
        storage_context (StorageContext): storage context containing
            docstore, index store and vector store.
        index_id (Optional[str]): ID of the index to load.
            Defaults to None, which assumes there's only a single index
            in the index store and load it.
        **kwargs: Additional keyword args to pass to the index constructors.
    """
    index_ids: Optional[Sequence[str]]
    if index_id is None:
        index_ids = None
    else:
        index_ids = [index_id]

    indices = load_indices_from_storage(storage_context, index_ids=index_ids, **kwargs)

    if len(indices) == 0:
        raise ValueError(
            "No index in storage context, check if you specified the right persist_dir."
        )
    elif len(indices) > 1:
        raise ValueError(
            f"Expected to load a single index, but got {len(indices)} instead. "
            "Please specify index_id."
        )

    return indices[0]


def index_from_index_struct(index_struct, storage_context, **kwargs) -> BaseIndex:
    logger.debug(f"index_from_index_struct: {index_struct.index_id}")
    type_ = index_struct.get_type()
    index_cls = INDEX_STRUCT_TYPE_TO_INDEX_CLASS[type_]
    object_map = reconstruct_object_map(index_struct, storage_context=storage_context, **kwargs)
    index = index_cls(
        index_struct=index_struct, storage_context=storage_context, **kwargs
    )
    index._object_map = object_map  # Add a conditional here.
    return index


def reconstruct_object_map(index_struct: IS, storage_context: StorageContext, **kwargs) -> Dict[str, Any]:
    logger.debug(f"reconstruct_object_map: {index_struct.index_id}")
    if isinstance(index_struct, IndexDict):
        nodes_dict = index_struct.nodes_dict
        node_ids = nodes_dict.keys()
    elif isinstance(index_struct, IndexList):
        node_ids = index_struct.nodes
    elif isinstance(index_struct, KeywordTable):
        node_ids = index_struct.node_ids
    elif isinstance(index_struct, KG):
        node_ids = index_struct.node_ids
    elif isinstance(index_struct, IndexGraph):
        node_ids = index_struct.all_nodes
    else:
        logger.warning(f"Unable to build mapping for {index_struct.index_id} of type {type(index_struct)}")
        node_ids = []

    object_map = {}
    for node_id in node_ids:
        node = storage_context.docstore.get_node(node_id)
        if not isinstance(node, IndexNode) or not node.index_id or node.index_id == index_struct.index_id:
            continue

        child_index_struct = storage_context.index_store.get_index_struct(node.index_id)
        if child_index_struct:
            obj = index_from_index_struct(child_index_struct, storage_context=storage_context, **kwargs)
            if obj is not None:
                object_map[node.index_id] = obj

    return object_map


def load_indices_from_storage(
        storage_context: StorageContext,
        index_ids: Optional[Sequence[str]] = None,
        **kwargs: Any,
) -> List[BaseIndex]:
    """Load multiple indices from storage context.

    Args:
        storage_context (StorageContext): storage context containing
            docstore, index store and vector store.
        index_id (Optional[Sequence[str]]): IDs of the indices to load.
            Defaults to None, which loads all indices in the index store.
        **kwargs: Additional keyword args to pass to the index constructors.
    """
    if index_ids is None:
        logger.info("Loading all indices.")
        index_structs = storage_context.index_store.index_structs()
    else:
        logger.info(f"Loading indices with ids: {index_ids}")
        index_structs = []
        for index_id in index_ids:
            index_struct = storage_context.index_store.get_index_struct(index_id)
            if index_struct is None:
                raise ValueError(f"Failed to load index with ID {index_id}")
            index_structs.append(index_struct)

    indices = []
    for index_struct in index_structs:
        index = index_from_index_struct(index_struct, storage_context, **kwargs)
        indices.append(index)
    return indices


def load_graph_from_storage(
        storage_context: StorageContext,
        root_id: str,
        **kwargs: Any,
) -> ComposableGraph:
    """Load composable graph from storage context.

    Args:
        storage_context (StorageContext): storage context containing
            docstore, index store and vector store.
        root_id (str): ID of the root index of the graph.
        **kwargs: Additional keyword args to pass to the index constructors.
    """
    indices = load_indices_from_storage(storage_context, index_ids=None, **kwargs)
    all_indices = {index.index_id: index for index in indices}
    return ComposableGraph(all_indices=all_indices, root_id=root_id)
