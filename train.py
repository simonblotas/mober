import torch
from loss import loss_function_vae, loss_function_classification
import wandb
from datetime import datetime
from metrics import project_into_decoded_space, metrics_on_dataloader
import numpy as np


def evaluate_model(model_BatchAE, model_src_adv, dataloader, device, parameters):
    model_BatchAE.eval()
    model_src_adv.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            genes, labels = batch[2].to(device), batch[0].to(device)
            dec, enc, means, stdev = model_BatchAE(genes, labels)
            v_loss = loss_function_vae(dec, genes, means, stdev, kl_weight=parameters['kl_weight'])

            src_pred = model_src_adv(enc)
            loss_src_adv = loss_function_classification(src_pred, labels, parameters['src_weights_src_adv'].to(device))

            loss_ae = v_loss - parameters['src_adv_weight'] * loss_src_adv
            total_loss += loss_ae.item() * genes.size(0)
            total_samples += genes.size(0)
    
    avg_loss = total_loss / total_samples
    return avg_loss




def train_model(model_BatchAE, 
                optimizer_BatchAE, 
                model_src_adv, 
                optimizer_src_adv,
                device,
                train_loader,
                valid_loader,
                test_loader,
                ach_features, 
                ach_abbreviations_labels,
                tcga_features, 
                tcga_abbreviations_labels,
                parameters):



    tcga_features_contiguous = np.ascontiguousarray(tcga_features)
    ach_features_contiguous = np.ascontiguousarray(ach_features)

    # Initialize WandB run


    # Get the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a dynamic project name using the current date and time
    traning_name = f"training_run_{current_time}"
    wandb.init(
        project="Mober",
        name=traning_name,
        config= {
        "vae_learning_rate": parameters['vae_learning_rate'],
        "adv_learning_rate": parameters['adv_learning_rate'],
        "n_genes": parameters['n_genes'],
        "encoded_dim": parameters['encoded_dim'],
        "n_sources": parameters['n_sources'],
        "epochs": parameters['epochs'],
        "kl_weight": parameters['kl_weight'],
        "early_stopping_patience": parameters['early_stopping_patience'],
        "src_adv_weight" : parameters['src_adv_weight'],
        "src_weights_src_adv" : parameters['src_weights_src_adv'].tolist(),
        'train_ratio' : parameters['train_ratio'],
        'val_ratio' : parameters['val_ratio'],
        'test_ratio' : parameters['test_ratio'],
        'Datasets' : parameters['Datasets'],
        'space_to_project_into' : parameters['space_to_project_into'],
        'K': parameters['K'], #Number of Nearest Neighboors considered 
        'enc_hidden_layers': parameters['enc_hidden_layers'],
        'enc_dropouts': parameters['enc_dropouts'],
        'dec_hidden_layers' : parameters['dec_hidden_layers'], 
        'dec_dropouts' : parameters['dec_dropouts'],
        'mlp_hidden_layers' : parameters['mlp_hidden_layers'],
        'mlp_dropouts': parameters['mlp_dropouts']
    })

    # Early stopping settings
    best_model_loss = float('inf')
    waited_epochs = 0
    early_stop = False
    
    for epoch in range(parameters['epochs']):
        if early_stop:
            break
        
        epoch_ae_loss = 0.0
        epoch_src_adv_loss = 0.0
        epoch_tot_loss = 0.0 
        
        model_BatchAE.train()
        model_src_adv.train()
        for batch in train_loader:
            genes, labels = batch[2].to(device), batch[0].to(device)

            dec, enc, means, stdev = model_BatchAE(genes, labels)
            v_loss = loss_function_vae(dec, genes, means, stdev, kl_weight=parameters['kl_weight'])

            # Source adversary
            model_src_adv.zero_grad()

            src_pred = model_src_adv(enc)

            loss_src_adv = loss_function_classification(src_pred, labels, parameters['src_weights_src_adv'].to(device))
            loss_src_adv.backward(retain_graph=True)
            epoch_src_adv_loss += loss_src_adv.detach().item()
            optimizer_src_adv.step()

            src_pred = model_src_adv(enc)
            loss_src_adv = loss_function_classification(src_pred, labels, parameters['src_weights_src_adv'].to(device))

            # Update ae
            model_BatchAE.zero_grad()
            loss_ae = v_loss - parameters['src_adv_weight'] * loss_src_adv
            loss_ae.backward()
            epoch_ae_loss += v_loss.detach().item()
            optimizer_BatchAE.step()
            
            epoch_tot_loss += loss_ae.detach().item()
            
        
        avg_train_ae_loss = epoch_ae_loss / len(train_loader.dataset)
        avg_train_adv_loss = epoch_src_adv_loss / len(train_loader.dataset)
        avg_train_tot_loss = epoch_tot_loss / len(train_loader.dataset)
        
        # Log metrics to WandB
        wandb.log({
            'Train/AE_Loss': avg_train_ae_loss, 
            'Train/Adv_Loss': avg_train_adv_loss, 
            'Train/Total_Loss': avg_train_tot_loss
        }, step=epoch)
        
        print(f"Epoch {epoch + 1}/{parameters['epochs']} - Train AE Loss: {avg_train_ae_loss:.4f}, "
              f"Train Adv Loss: {avg_train_adv_loss:.4f}, Train Total Loss: {avg_train_tot_loss:.4f}")


        # Metric on training set :
        train, train_labels =  project_into_decoded_space(train_loader, model_BatchAE,device,parameters['space_to_project_into'])
        if parameters['space_to_project_into'] == 'TCGA':
            val_acc = metrics_on_dataloader(tcga_abbreviations_labels.numpy(), tcga_features_contiguous, train_labels, train)
        elif parameters['space_to_project_into'] == 'ACH':
            val_acc = metrics_on_dataloader(ach_abbreviations_labels.numpy(), ach_features_contiguous, train_labels, train)
        wandb.log({'Train_Accuracy': val_acc}, step=epoch)
        

        # Validation
        if valid_loader is not None:
            avg_valid_loss = evaluate_model(model_BatchAE, model_src_adv, valid_loader, device, parameters)
            wandb.log({'Validation/Total_Loss': avg_valid_loss}, step=epoch)
            

            # Early stopping
            if avg_valid_loss < best_model_loss:
                best_model_loss = avg_valid_loss
                waited_epochs = 0
            else:
                waited_epochs += 1
                if waited_epochs >= parameters['early_stopping_patience']:
                    early_stop = True
                    print("Early stopping triggered.")
                        
            val, val_labels =  project_into_decoded_space(valid_loader, model_BatchAE,device,parameters['space_to_project_into'])
            if parameters['space_to_project_into'] == 'TCGA':
                tcga_abbreviations_labels.to(torch.float32)
                val_acc = metrics_on_dataloader(tcga_abbreviations_labels.numpy(), tcga_features_contiguous, val_labels, val)
            elif parameters['space_to_project_into'] == 'ACH':
                val_acc = metrics_on_dataloader(ach_abbreviations_labels.numpy(), ach_features_contiguous, val_labels, val)
            wandb.log({'Validation_Accuracy': val_acc}, step=epoch)
        
        if test_loader is not None:
            avg_test_loss = evaluate_model(model_BatchAE, model_src_adv, test_loader, device, parameters)
            wandb.log({'Test/Total_Loss': avg_test_loss}, step=epoch)
            test, test_labels = project_into_decoded_space(test_loader, model_BatchAE,device,parameters['space_to_project_into'])
            if parameters['space_to_project_into'] == 'TCGA':
                test_acc = metrics_on_dataloader(tcga_abbreviations_labels.numpy(), tcga_features_contiguous, test_labels, test)
            elif parameters['space_to_project_into'] == 'ACH':
                test_acc = metrics_on_dataloader(ach_abbreviations_labels.numpy(), ach_features_contiguous, test_labels, test)
            wandb.log({'Test_Accuracy': test_acc}, step=epoch)
    
    wandb.finish()  # Finish WandB run

