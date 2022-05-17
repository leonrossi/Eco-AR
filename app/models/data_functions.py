import app
import numpy as np
import pandas as pd
from app.models.mtree import mtree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dataclasses import dataclass
import random
# import dash_table

@dataclass
class DataInfo:
    df: pd.DataFrame
    df_norm: pd.DataFrame
    features: str
    components: float
    data_struct: mtree.MTree
    isCategorized: bool
    fPCA: bool
    fTSNE: bool

eventinfo = DataInfo(None, None, None, None, None, None, None, None) 

def data_normalization():
    if(eventinfo.isCategorized): audio_features = eventinfo.features[1:-1]
    else: audio_features = eventinfo.features[1:]
    
    df_norm = pd.DataFrame(columns=audio_features)
    for feat in audio_features:
        df_norm[feat] = (eventinfo.df[feat]-eventinfo.df[feat].min())/(eventinfo.df[feat].max()-eventinfo.df[feat].min())

    return df_norm

def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

def get_data_information(datafile):
    df = pd.read_csv(datafile)
    features = []
    [features.append(col) for col in df.columns] # obtendo nomes das colunas 

    eventinfo.df = df
    eventinfo.features = features
    if(features[-1] == 'label'): eventinfo.isCategorized = True
    else: eventinfo.isCategorized = False
    eventinfo.df_norm = data_normalization()

    return eventinfo

def get_data_projection(eventinfo, PCAflag, TSNEflag):
    if(eventinfo.isCategorized): audio_features = eventinfo.features[1:-1]
    else: audio_features = eventinfo.features[1:]
    
    random.seed(42)
    np.random.seed(42)
    if(PCAflag):
        pca = PCA(n_components=2)
        components = pca.fit_transform(eventinfo.df_norm[audio_features])
    elif(TSNEflag):
        tsne = TSNE(n_components=2)
        components = tsne.fit_transform(eventinfo.df_norm[audio_features])

    eventinfo.fPCA = PCAflag
    eventinfo.fTSNE = TSNEflag
    eventinfo.components = components
    
    return eventinfo

def index_data(eventinfo):
    if eventinfo is not None:
        result = np.array(eventinfo.df_norm)
        np.random.seed(42)
        random.seed(42)
        data_struct = mtree.MTree(euclidean_distance, max_node_size=4)
        for line in result:
            data_struct.add(line)

        eventinfo.data_struct = data_struct

    return eventinfo

def knn_search(eventinfo, guide_file, k):
    guide = pd.read_csv(guide_file)
    k = int(k)

    np.random.seed(42)
    random.seed(42)

    # Check if the guide is on the dataset
    guide_index = None
    for index, row in eventinfo.df.iterrows():
        aux = (row == guide).all()
        if(aux.all()):
            guide_index = index
            break

    # if it is, look for the pair in df_norm and assign it to guide variable
    # if not, add to the struct, generate the normalization, calculate the components and assign it to guide variable
    if(guide_index is None):
        eventinfo.df = eventinfo.df.append(guide, ignore_index=True) # new original dataset with the guide elemnt
        eventinfo.df_norm = data_normalization() # get normalization
        eventinfo = index_data(eventinfo) # create data_struct again with new element
        eventinfo = get_data_projection(eventinfo, eventinfo.fPCA, eventinfo.fTSNE) # generate the components
        guide = np.array(eventinfo.df_norm)[-1]
    else:
        guide = np.array(eventinfo.df_norm)[guide_index]

    # Doing the knn search
    retrieval = eventinfo.data_struct.search(guide, k)

    list_retrieval = list(retrieval); isNone = True
    # Clearing None values (structure problem)
    while isNone:
        if(list_retrieval[-1] is None):
            list_retrieval = list_retrieval[:-1]
        else: isNone = False

    # Search the index of the retrieval elements
    retrieval_index = []
    # audio_features = eventinfo.features[1:-1]
    if(eventinfo.isCategorized): audio_features = eventinfo.features[1:-1]
    else: audio_features = eventinfo.features[1:]
    df_aux = pd.DataFrame(list_retrieval, columns=audio_features)
    for index, row in df_aux.iterrows():
        for index_event, row_event in eventinfo.df_norm.iterrows():
            aux = (row == row_event).all()
            if(aux.all()):
                retrieval_index.append(index_event)

    retrieval_components = np.arange(len(retrieval_index)*2, dtype=float).reshape(len(retrieval_index),2)
    for i in range(len(retrieval_index)):
        temp = eventinfo.components[retrieval_index[i],:]
        retrieval_components[i,0] = temp[0]
        retrieval_components[i,1] = temp[1]

    # print(f"INDEX: ", retrieval_index)
    # retrieval_files = []
    # for i in retrieval_index:
    #     retrieval_files.append(eventinfo.df[i]['filename'])
    # print(f"FILES: ", retrieval_files)

    return eventinfo, retrieval_components

# def create_data_table():
#     """Create Dash datatable from Pandas DataFrame."""
#     table = dash_table.DataTable(
#         id='database-table',
#         columns=[{"name": i, "id": i} for i in eventinfo.df.columns],
#         data=eventinfo.df.to_dict('records'),
#         sort_action="native",
#         sort_mode='native',
#         page_size=300
#     )
#     return table