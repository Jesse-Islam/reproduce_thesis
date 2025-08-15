from gosip import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import anndata as ad
import shutil
import random

from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import unittest
import traceback
from sklearn.preprocessing import MinMaxScaler


class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        try:
            num_epochs=10
            torch.manual_seed(1)
            np.random.seed(2)
            random.seed(3)
            sns.set(style="whitegrid")
            main_path=""
            category_labels=["cell_type"]
            validation_ratio=0.1

            single_cell_data="testing"
            outdir=single_cell_data.replace(" ", "_")
            if not os.path.exists(outdir):
                os.makedirs(outdir)



            print(ad.__version__)
            #adapted from  https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html
            # Define the desired correlation matrix
            corr_matrix = np.array([
                [1.0, -0.8, -0.6, -0.4, -0.2],
                [-0.8, 1.0, -0.7, -0.5, -0.3],
                [-0.6, -0.7, 1.0, 0.6, 0.4],
                [-0.4, -0.5, 0.6, 1.0, 0.5],
                [-0.2, -0.3, 0.4, 0.5, 1.0]
            ])
            # Generate random data with the desired correlation matrix
            sample_size=500
            counts_B = np.random.multivariate_normal(mean=[2, 5, -8, 10, -15], cov=corr_matrix, size=sample_size)
            counts_T = np.random.multivariate_normal(mean=[0, 0, 8, 0, 15], cov=corr_matrix, size=sample_size)
            counts_M = np.random.multivariate_normal(mean=[4, -2, 0, -5, 0], cov=corr_matrix, size=sample_size)
            counts=np.concatenate((counts_B, counts_T, counts_M), axis=0)
            # Repeat values for each category
            repeated_B = np.repeat("B", sample_size)
            repeated_T = np.repeat("T", sample_size)
            repeated_M = np.repeat("M", sample_size)
            # Create a NumPy array with the repeated values
            ct= np.concatenate((repeated_B, repeated_T, repeated_M),axis=0)
            print(ct.shape)
            print(counts.shape)
            adata = ad.AnnData(counts)
            adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
            adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
            adata.obs["cell_type"] = pd.Categorical(ct)  # Categoricals are preferred for efficiency

            
            # Assuming `adata` is your AnnData object
            scaler = MinMaxScaler()
            
            # Scale adata.X
            adata.X = scaler.fit_transform(adata.X)
            adata.X = (adata.X-adata.X.min())/(adata.X.max()-adata.X.min())
            one_hot_labels, num_categories= one_hot_encode_combinations(adata.obs, category_labels)
            adata.one_hot_labels= one_hot_labels
            device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


            stargan_hyperparam = {
                'batch_size': 100,
                'learning_rate': 0.1,
                'layer_nodes_generator': [10,10,10],
                'layer_nodes_discriminator': [10,10,10],
                'critics': 1,
                "dropout":0.1
            }
            start_time=time.time()
            stargan = StarGAN(input_dim=adata.shape[1],
                             num_domains=[adata.one_hot_labels.shape[1]],
                             device = device,
                             learning_rate = stargan_hyperparam['learning_rate'],
                             layer_g = stargan_hyperparam['layer_nodes_generator'],
                             layer_d = stargan_hyperparam['layer_nodes_discriminator'],
                             critics = stargan_hyperparam['critics'],
                             lambda_rec=10,
                             #lambda_iden=10,
                              dropout_rate=stargan_hyperparam['dropout'],zi=False)

            # Assume 'adata' is your AnnData object
            indices = np.arange(adata.n_obs)

            # Split indices into training and validation sets
            train_idx, val_idx = train_test_split(indices, test_size=validation_ratio, random_state=42)

            # Subset the AnnData object to create training and validation sets
            train_adata = adata[train_idx, :].copy()
            train_adata.one_hot_labels=adata.one_hot_labels.iloc[train_idx,:].copy()
            val_adata = adata[val_idx, :].copy()
            val_adata.one_hot_labels=adata.one_hot_labels.iloc[val_idx,:].copy()
            train_dataloader = prepare_data(train_adata,num_categories,batch_size=stargan_hyperparam["batch_size"])
            val_dataloader = prepare_data(val_adata,num_categories,batch_size=stargan_hyperparam["batch_size"])


            stargan.train(dataloader=train_dataloader,val_loader=val_dataloader, num_epochs=num_epochs,burn_in=2,verbose=True)
            stargan.G.eval()
            stargan.D.eval()
            torch.cuda.empty_cache()
            print(time.time()-start_time)


            oracle_hyperparam = {
                'batch_size': 10,
                'learning_rate': 0.1,
                'layer_nodes': [10,10,10],
                'dropout_rate_d': 0.001
            }


            start_time=time.time()
            #adata.one_hot_labels.shape[1]
            oracle = Oracle(input_dim=adata.shape[1],
                             num_domains=num_categories,
                             device = device,
                             learning_rate = oracle_hyperparam['learning_rate'],
                             layer_d = oracle_hyperparam['layer_nodes'],
                             drpt_d = oracle_hyperparam['dropout_rate_d'])


            oracle.train(dataloader=train_dataloader, val_loader=val_dataloader, num_epochs=num_epochs, verbose=False,optuna_run=False)
            oracle.D.eval()
            torch.cuda.empty_cache()
            print(time.time()-start_time)

            propagator_hyperparam = {
                'batch_size': 10,
                'learning_rate': 0.1,
                'layer_nodes_generator': [10,10,10],
                'dropout_rate_g': 0.001,
                'latent_dim':10,
                'beta':6.0
            }


            start_time=time.time()
            #adata.one_hot_labels.shape[1]


    
            # Now you can use `train_dataloader` and `val_dataloader` for training your model.

            train_dataloader = prepare_data(train_adata,num_categories,batch_size=stargan_hyperparam["batch_size"])
            val_dataloader = prepare_data(val_adata,num_categories,batch_size=stargan_hyperparam["batch_size"])
            propagator = Propagator(input_dim=train_adata.shape[1],
                             num_domains=num_categories,
                             device = device,
                             learning_rate = propagator_hyperparam['learning_rate'],
                             layer_g = propagator_hyperparam['layer_nodes_generator'],
                             drpt_g = propagator_hyperparam['dropout_rate_g'],
                             latent_dim=propagator_hyperparam['latent_dim'],zi=False)

            #loss_fn = BetaVaeLoss(beta=1.0)
            loss_fn =  BtcvaeLoss(n_data=train_adata.shape[0],beta=1.0,zi=False)
            propagator.train(dataloader=train_dataloader, val_loader=val_dataloader,
                             num_epochs=num_epochs, verbose=True,optuna_run=False,
                             loss_fn=loss_fn)
            propagator.G.eval()
            torch.cuda.empty_cache()
            print(time.time()-start_time)
            shared_filter = None
            filter_criteria_start = ["M"] #["cell_disease=beta cell_Tumor"] #
            filter_criteria_goal =["B"] # ["cell_disease=beta cell_Normal"]#


            perturbation_metrics,accuracy,result_path=full_report(adata=adata,
                                                      stargan=stargan,
                                                      oracle=oracle,
                                                      propagator=propagator,
                                                      shared_filter=shared_filter,
                                                      filter_criteria_start=filter_criteria_start,
                                                      filter_criteria_goal=filter_criteria_goal,
                                                      main_path=main_path,outdir=outdir,top_n=20,percentile=0,
                                                      umap=True,oracle_performance=True,alpha=0.5,
                                                      sample_fraction=1,
                                                      num_categories=num_categories,
                                                      category_labels=category_labels,
                                                      device = device,#k=2,
                                                                 )
            
            print(perturbation_metrics.head())

            plot_volcano(perturbation_metrics,"suggested_perturbation",'oracle_score',0,0.001,labels=True,left_label="strong decrease",right_label="strong increase",title='importance vs. foldchange')
            plt.savefig(result_path+ "/" +"suggested_perturbation.png")

            plot_volcano_one_sided(perturbation_metrics,"suggested_perturbation",'oracle_score',0,0.002,labels=True,left_label="Negatively influenced",right_label="Positively influenced",title="In degree influence")
            plt.savefig(result_path+ "/" +"testing_one_sided_volcano.png")
            shutil.rmtree(single_cell_data)
            
            stargan = StarGAN(input_dim=adata.shape[1],
                             num_domains=[adata.one_hot_labels.shape[1]],
                             device = device,
                             learning_rate = stargan_hyperparam['learning_rate'],
                             layer_g = stargan_hyperparam['layer_nodes_generator'],
                             layer_d = stargan_hyperparam['layer_nodes_discriminator'],
                             critics = stargan_hyperparam['critics'],
                             lambda_rec=10,
                             #lambda_iden=10,
                              dropout_rate=stargan_hyperparam['dropout'],zi=True)

            # Assume 'adata' is your AnnData object
            indices = np.arange(adata.n_obs)

            # Split indices into training and validation sets
            train_idx, val_idx = train_test_split(indices, test_size=validation_ratio, random_state=42)

            # Subset the AnnData object to create training and validation sets
            train_adata = adata[train_idx, :].copy()
            train_adata.one_hot_labels=adata.one_hot_labels.iloc[train_idx,:].copy()
            val_adata = adata[val_idx, :].copy()
            val_adata.one_hot_labels=adata.one_hot_labels.iloc[val_idx,:].copy()
            train_dataloader = prepare_data(train_adata,num_categories,batch_size=stargan_hyperparam["batch_size"])
            val_dataloader = prepare_data(val_adata,num_categories,batch_size=stargan_hyperparam["batch_size"])


            stargan.train(dataloader=train_dataloader,val_loader=val_dataloader, num_epochs=num_epochs,burn_in=2,verbose=True)
            stargan.G.eval()
            stargan.D.eval()
            torch.cuda.empty_cache()
            print(time.time()-start_time)


            propagator_hyperparam = {
                'batch_size': 10,
                'learning_rate': 0.1,
                'layer_nodes_generator': [10,10,10],
                'dropout_rate_g': 0.001,
                'latent_dim':10,
                'beta':6.0
            }


            start_time=time.time()
            #adata.one_hot_labels.shape[1]


    
            # Now you can use `train_dataloader` and `val_dataloader` for training your model.

            train_dataloader = prepare_data(train_adata,num_categories,batch_size=stargan_hyperparam["batch_size"])
            val_dataloader = prepare_data(val_adata,num_categories,batch_size=stargan_hyperparam["batch_size"])
            propagator = Propagator(input_dim=train_adata.shape[1],
                             num_domains=num_categories,
                             device = device,
                             learning_rate = propagator_hyperparam['learning_rate'],
                             layer_g = propagator_hyperparam['layer_nodes_generator'],
                             drpt_g = propagator_hyperparam['dropout_rate_g'],
                             latent_dim=propagator_hyperparam['latent_dim'],zi=True)

            #loss_fn = BetaVaeLoss(beta=1.0)
            loss_fn =  BtcvaeLoss(n_data=train_adata.shape[0],beta=1.0,zi=True)
            propagator.train(dataloader=train_dataloader, val_loader=val_dataloader,
                             num_epochs=num_epochs, verbose=True,optuna_run=False,
                             loss_fn=loss_fn)
            propagator.G.eval()
            torch.cuda.empty_cache()
            print(time.time()-start_time)
            shared_filter = None
            filter_criteria_start = ["M"] #["cell_disease=beta cell_Tumor"] #
            filter_criteria_goal =["B"] # ["cell_disease=beta cell_Normal"]#


            perturbation_metrics,accuracy,result_path=full_report(adata=adata,
                                                      stargan=stargan,
                                                      oracle=oracle,
                                                      propagator=propagator,
                                                      shared_filter=shared_filter,
                                                      filter_criteria_start=filter_criteria_start,
                                                      filter_criteria_goal=filter_criteria_goal,
                                                      main_path=main_path,outdir=outdir,top_n=20,percentile=0,
                                                      umap=True,oracle_performance=True,alpha=0.5,
                                                      sample_fraction=1,
                                                      num_categories=num_categories,
                                                      category_labels=category_labels,
                                                      device = device,#k=2,
                                                                 )
            
            print(perturbation_metrics.head())

            plot_volcano(perturbation_metrics,"suggested_perturbation",'oracle_score',0,0.001,labels=True,left_label="strong decrease",right_label="strong increase",title='importance vs. foldchange')
            plt.savefig(result_path+ "/" +"suggested_perturbation.png")

            plot_volcano_one_sided(perturbation_metrics,"suggested_perturbation",'oracle_score',0,0.002,labels=True,left_label="Negatively influenced",right_label="Positively influenced",title="In degree influence")
            plt.savefig(result_path+ "/" +"testing_one_sided_volcano.png")
            shutil.rmtree(single_cell_data)

            
        except Exception as e:
            # Capture the stack trace, including the line where the error occurred
            tb = traceback.format_exc()
            
            # Create a detailed error message
            error_message = f'The pipeline failed with exception: {str(e)}\nTraceback:\n{tb}'
            
            # Log or handle the failure (e.g., self.fail can be customized to log this appropriately)
            self.fail(error_message)
        

if __name__ == '__main__':
    unittest.main()
