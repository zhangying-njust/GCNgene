# 聚类，泊松分布的负对数似然来计算损失
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.io
import anndata as ad
import argparse
from sklearn.metrics import pairwise_distances
import torch
from MyModel import DeconvNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from Calculate import CalculateMeteics
def adata_to_cluster_expression(adata, cluster_label, scale=True, add_density=True):
    try:
        value_counts = adata.obs[cluster_label].value_counts(normalize=True)
    except KeyError as e:
        raise ValueError("Provided label must belong to adata.obs.")
    unique_labels = value_counts.index
    new_obs = pd.DataFrame({cluster_label: unique_labels})
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)

    X_new = np.empty((len(unique_labels), adata.shape[1]))
    for index, l in enumerate(unique_labels):
        if not scale:
            X_new[index] = adata[adata.obs[cluster_label] == l].X.mean(axis=0)
        else:
            X_new[index] = adata[adata.obs[cluster_label] == l].X.sum(axis=0)


    adata_ret.X = X_new

    if add_density:
        adata_ret.obs["cluster_density"] = adata_ret.obs[cluster_label].map(
            lambda i: value_counts[i]
        )

    return adata_ret

def select_hvgs(adata_ref, celltype_ref_col, num_per_group=200):
    sc.tl.rank_genes_groups(adata_ref, groupby=celltype_ref_col, method="t-test", key_added="ttest", use_raw=False)
    markers_df = pd.DataFrame(adata_ref.uns['ttest']['names']).iloc[0:num_per_group, :]
    genes = sorted(list(np.unique(markers_df.melt().value.values)))
    return genes

def build_graph(adata_st_input,  # list of spatial transcriptomics (ST) anndata objects
                   adata_ref_input,  # reference single-cell anndata object
                   KFold,
                   celltype_ref_col="celltype",  # column of adata_ref_input.obs for cell type information
                   sample_col=None,  # column of adata_ref_input.obs for batch labels
                   celltype_ref=None,  # specify cell types to use for deconvolution
                   n_hvg_group=500,  # number of highly variable genes for reference anndata
                   three_dim_coor=None,  # if not None, use existing 3d coordinates in shape [# of total spots, 3]
                   coor_key="spatial_aligned",  # "spatial_aligned" by default
                   rad_cutoff=None,  # cutoff radius of spots for building graph
                   rad_coef=1.1,
                   # if rad_cutoff=None, rad_cutoff is the minimum distance between spots multiplies rad_coef
                   prune_graph_cos=False,  # prune graph connections according to cosine similarity
                   cos_threshold=0.5,  # threshold for pruning graph connections
                   c2c_dist=100,  # center to center distance between nearest spots in micrometer
                   cluster=True,
                   ):


    test_list = predict_gene[KFold]
    train_list = train_gene[KFold]

    print("Finding highly variable genes...")
    adata_ref = adata_ref_input.copy()
    adata_ref.var_names_make_unique()
    # # Remove mt-genes
    # adata_ref = adata_ref[:, np.array(~adata_ref.var.index.isna())
    #                       & np.array(~adata_ref.var_names.str.startswith("mt-"))
    #                       & np.array(~adata_ref.var_names.str.startswith("MT-"))]
    if celltype_ref is not None:
        if not isinstance(celltype_ref, list):
            raise ValueError("'celltype_ref' must be a list!")
        else:
            adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]
    else:
        celltype_counts = adata_ref.obs[celltype_ref_col].value_counts()
        celltype_ref = list(celltype_counts.index[celltype_counts > 1])
        adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]

    # Remove cells and genes with 0 counts
    sc.pp.filter_cells(adata_ref, min_genes=1)
    # =====================================================过滤后，参考组会丢失一些基因，导致测试集中的某些无法预测，因此不过滤
    # sc.pp.filter_genes(adata_ref, min_cells=1)

    # Concatenate ST adatas
    adata_st_new = adata_st_input.copy()
    adata_st_new.var_names_make_unique()
    # Remove mt-genes
    # adata_st_new = adata_st_new[:, (np.array(~adata_st_new.var.index.str.startswith("mt-"))
    #                                 & np.array(~adata_st_new.var.index.str.startswith("MT-")))]
    adata_st = adata_st_new

    print("Calculate basis for deconvolution...")

    if cluster:
        adata_basis = adata_to_cluster_expression(adata_ref, 'celltype', scale=False, add_density=True)
    else:
        adata_basis = adata_ref.copy()

    # Take gene intersection
    genes = list(adata_st.var.index & adata_ref.var.index & train_list)
    adata_ref_train = adata_ref[:, genes]
    adata_st_train = adata_st[:, genes]
    adata_basis_train = adata_basis[:, genes]

    print("Preprocess ST data...")
    st_mtx = adata_st_train.X.copy()
    if scipy.sparse.issparse(st_mtx):
        st_mtx = st_mtx.toarray()
    adata_st_train.obsm["count"] = st_mtx
    st_library_size = np.sum(st_mtx, axis=1)
    adata_st_train.obs["library_size"] = st_library_size
    if scipy.sparse.issparse(adata_st_train.X):
        adata_st_train.X = adata_st_train.X.toarray()

    # Build a graph for spots across multiple slices
    print("Start building a graph...")

    # Build 3D coordinates
    if three_dim_coor is None:

        adata_st_ref = adata_st_train.copy()

        # loc_ref = np.array(adata_st_ref.obsm[coor_key])
        loc = np.vstack((adata_st_ref.obs['array_row'].values, adata_st_ref.obs['array_col'].values)).T

        pair_dist_ref = pairwise_distances(loc)
        min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]

        if rad_cutoff is None:
            # The radius is computed base on the attribute "adata.obsm['spatial']"
            rad_cutoff = min_dist_ref * rad_coef
        print("Radius for graph connection is %.4f." % rad_cutoff)

    # If 3D coordinates already exists
    else:
        if rad_cutoff is None:
            raise ValueError("Please specify 'rad_cutoff' for finding 3D neighbors!")
        loc = three_dim_coor

    pair_dist = pairwise_distances(loc)
    G = (pair_dist < rad_cutoff).astype(float)

    if prune_graph_cos:
        pair_dist_cos = pairwise_distances(adata_st_ref.X, metric="cosine") # 1 - cosine_similarity
        G_cos = (pair_dist_cos < (1 - cos_threshold)).astype(float)
        G = G * G_cos

    print('%.4f neighbors per cell on average.' % (np.mean(np.sum(G, axis=1)) - 1))
    adata_st_train.obsm["graph"] = G
    adata_st_train.obsm["3D_coor"] = loc

    # return adata_st, adata_basis
    return adata_st, adata_ref, adata_basis, adata_st_train, adata_ref_train, adata_basis_train



def train(adata_st, adata_basis, model, device, kFold, save_path, total_epoch=5000, report_loss=True, step_interval=500):
    # prepare data
    if scipy.sparse.issparse(adata_st.X):
        X = torch.from_numpy(adata_st.X.toarray()).float().to(device)
    else:
        X = torch.from_numpy(adata_st.X).float().to(device)
    A = torch.from_numpy(np.array(adata_st.obsm["graph"])).float().to(device)
    Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(device)
    lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(device)
    basis = torch.from_numpy(np.array(adata_basis.X)).float().to(device)

    # setting
    lr = 2e-3
    optimizer = optim.Adamax(list(model.parameters()), lr=lr)

    model.to(device)
    for step in tqdm(range(total_epoch)):
        model.train()
        loss = model(adj_matrix=A,
                    node_feats=X,
                    count_matrix=Y,
                    library_size=lY,
                    basis=basis)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if report_loss:
            if not step % step_interval:
                print("Step: %s, Loss: %.4f, d_loss_1: %.4f, d_loss_2: %.4f, f_loss: %.4f" % (
                    step, loss.item(), model.decon_loss_1.item(), model.decon_loss_2.item(), model.features_loss.item()))
    model_save_path = save_path + "save_model" + str(kFold) + ".pkl"
    torch.save(model.state_dict(), model_save_path)


    return

def val(adata_st, adata_basis,kFold, model, device):
    # prepare data
    if scipy.sparse.issparse(adata_st.X):
        X = torch.from_numpy(adata_st.X.toarray()).float().to(device)
    else:
        X = torch.from_numpy(adata_st.X).float().to(device)
    A = torch.from_numpy(np.array(adata_st.obsm["graph"])).float().to(device)
    Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(device)
    lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(device)
    basis = torch.from_numpy(np.array(adata_basis.X)).float().to(device)

    model.eval()
    model.to(device)
    Z, beta, p = model.evaluate(A, X)

    log_lam = torch.log(torch.matmul(beta, basis) + 1e-6)
    # log_lam = torch.log(torch.matmul(beta, basis) + 1e-6)  + self.gamma[slice_label]
    lam = torch.exp(log_lam)
    impute = p * lY * lam

    adata_ge = sc.AnnData(impute.cpu().detach().numpy())

    df_gene_ids = pd.DataFrame({"gene_ids": adata_basis.var.index})
    df_gene_ids = df_gene_ids.set_index("gene_ids")
    adata_ge.var = df_gene_ids

    test_list = predict_gene[kFold]
    genes = list(adata_ge.var.index & test_list)
    pre_gene = pd.DataFrame(adata_ge[:, test_list].X, columns=adata_ge[:, test_list].var.index)

    return pre_gene

def parse_args():
    # 'A549','brain','CD8T','HCT116','HEK293','HEK293T','HeLa','HepG2','kidney','liver','MOLM13'
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="../DataUpload/Dataset4/", type=str, help="The input data dir.", )
    parser.add_argument("--save_path", default="./output_results/", type=str, help="model_save_path", )
    parser.add_argument("--rate1", default=0.05, type=float, help="rate1", )
    parser.add_argument('--rate2', default=0.05, type=float, help="rate2",)
    parser.add_argument("--cuda_divices", default="1", type=str)
    return parser.parse_known_args()[0]

def _main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_divices
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global train_gene, predict_gene

    # 读入最原始数据
    RNA_file = args.datapath + 'scRNA_count.txt'
    Spatial_file = args.datapath + 'Insitu_count.txt'
    location_file = args.datapath + 'Locations.txt'

    adata_ref = sc.read(RNA_file, sep='\t', first_column_names=True).T
    adata_st = sc.read(Spatial_file, sep='\t')
    locations = np.loadtxt(location_file, skiprows=1)

    df_loc = pd.DataFrame({"array_row": locations[:, 0], "array_col": locations[:, 1]})
    adata_st.obs = df_loc
    df_gene_ids = pd.DataFrame({"gene_ids": adata_st.var.index})
    df_gene_ids = df_gene_ids.set_index("gene_ids")
    adata_st.var = df_gene_ids

    RNA_data_adata_label = adata_ref
    sc.pp.normalize_total(RNA_data_adata_label)
    sc.pp.log1p(RNA_data_adata_label)
    sc.pp.highly_variable_genes(RNA_data_adata_label)
    RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
    sc.pp.scale(RNA_data_adata_label, max_value=10)
    sc.tl.pca(RNA_data_adata_label)
    sc.pp.neighbors(RNA_data_adata_label)
    sc.tl.leiden(RNA_data_adata_label, resolution=0.5)
    adata_ref.obs = pd.DataFrame({"celltype": RNA_data_adata_label.obs.leiden})

    train_gene = np.load(args.datapath + 'train_list.npy', allow_pickle=True).tolist()
    predict_gene = np.load(args.datapath + 'test_list.npy', allow_pickle=True).tolist()


    impute_all = pd.DataFrame(columns=[])  # 定义空DataFrame
    for kFold in range(len(train_gene)):
        print("kFold:" + str(kFold) + " (from 0 to " + str(len(train_gene)-1) + ")")
        adata_st, adata_ref, adata_basis,\
        adata_st_train, adata_ref_train, adata_basis_train = build_graph(adata_st,
                                                                         adata_ref,
                                                                         kFold,
                                                                         rad_coef=5,
                                                                         n_hvg_group=500,
                                                                         cluster=True)

        save_path = args.save_path+args.datapath.split('/')[-2]+'/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        else:
            pass

        my_model = DeconvNet(hidden_dims=[adata_st_train.shape[1]] + [512, 128],
                             n_celltypes=adata_basis_train.shape[0],
                             n_heads=1,
                             rate1=args.rate1,
                             rate2=args.rate2)

        train(adata_st_train, adata_basis_train, my_model, device, kFold, save_path)
        result = val(adata_st_train, adata_basis,kFold, my_model, device)
        impute_all = pd.concat([impute_all, result], axis=1)

    impute_all.to_csv(save_path + "impute.csv", header=1, index=1)
    raw = pd.DataFrame(adata_st.X, columns=adata_st.var.index)

    prefix = args.save_path
    metric = ['PCC', 'SSIM', 'RMSE', 'JS']
    CM = CalculateMeteics(raw_count_file=raw, impute_count_file=impute_all, prefix=prefix,
                          metric=metric)

    result_all = CM.compute_all()
    result_mean = np.mean(result_all.values, axis=1)

    result_all.T.to_csv(save_path + "Metrics.txt", sep='\t', header=1, index=1)
    output_eval_file = save_path + "result_mean.txt"
    i=0
    with open(output_eval_file, "a") as writer:
        for key in list(result_all.index):
            result_rt = key + " " + str(result_mean[i])
            i=i+1
            writer.write(result_rt + "\n")


if __name__ == '__main__':
    _main()