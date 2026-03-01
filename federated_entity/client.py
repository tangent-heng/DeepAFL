from cProfile import label
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100

import dataset


class Muti_FeatureDataset(Dataset):
    def __init__(self, features, X_t, labels):
        self.features = features
        self.X_t = X_t
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.X_t[idx], self.labels[idx]


class FederatedClient:
    def __init__(self, client_id, backbone, train_set, device, args):
        self.client_id = client_id
        self.backbone = backbone.to(device)
        self.train_set = train_set
        self.device = device
        self.args = args

        self.train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

        feat_size = args.feat_size
        self.fe_size = args.fe if args.fe != 0 else feat_size
        self.hidden_dim = args.hidden_dim
        self.muti_feature = None

    def train(self, global_model, layer_idx, train_type="top_classifiers"):
        global_model_ = global_model

        # Training logic based on train_type
        if train_type == "top_classifiers":
            return self.train_top_layer(global_model_, layer_idx)
        elif train_type == "sandwiched_layers":
            return self.train_sandwiched_layers(global_model_, layer_idx)
        else:
            raise ValueError(f"Unknown train_type: {train_type}")


    def get_features(self, global_model):
        if self.muti_feature is not None:
            return self.muti_feature
        else:
            features = []
            features2 = []
            labels = []
            with torch.no_grad():
                for images, label in self.train_loader:
                    images = images.to(self.device)

                    feature = self.backbone(images)

                    features.append(feature.cpu())
                    # X_t.append(X.cpu())
                    labels.append(label)
            # if not global_model.is_fit_rf_0:
            #     global_model.rf_0.fit(torch.cat(features), torch.cat(labels))
            #     global_model.is_fit_rf_0 = True
            X_t = global_model.rf_0(torch.cat(features))
            X_t = X_t.cpu()


            feature_set = Muti_FeatureDataset(torch.cat(features), X_t, torch.cat(labels))

            self.muti_feature = DataLoader(
                feature_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.workers,
            )
            del self.backbone
            del self.train_loader
            del self.train_set

            return self.muti_feature

    def clear_features(self):
        self.muti_feature = None


    def train_top_layer(self, global_model, layer_idx):

        corr_X = torch.zeros(self.hidden_dim, self.hidden_dim).to(self.device)
        corr_Y = torch.zeros(self.hidden_dim, self.args.num_classes).to(self.device)

        feature = self.get_features(global_model)
        X_t = []

        # X, Y = feature.dataset.X_t, feature.dataset.labels
        # if layer_idx < self.args.n_layers and not global_model.is_fit_randfeat[layer_idx]:
        #     global_model.rf_layers[layer_idx].Ft.fit(X, Y)
        #     global_model.is_fit_randfeat[layer_idx] = True

        with torch.no_grad():
            for features, X, labels in feature:
                features = features.to(self.device)
                X = X.to(self.device)
                labels = labels.to(self.device)

                if layer_idx != 0:
                    F = global_model.rf_layers[layer_idx-1](X)
                    sandwiched_feature = global_model.sandwiched_layers[layer_idx-1](F)
                    X = X + sandwiched_feature
                    X_t.append(X.cpu())

                corr_X += torch.t(X) @ X
                corr_Y += torch.t(X) @ labels
        if layer_idx != 0:
            self.muti_feature.dataset.X_t = torch.cat(X_t)

        return corr_X, corr_Y

    def train_sandwiched_layers(self, global_model, layer_idx):
        features = self.get_features(global_model)
        global_model = global_model.to(self.device)
        FF = None
        WRF = None
        with torch.no_grad():
            for features, X, labels in features:
                X = X.to(self.device)
                labels = labels.to(self.device)
                F = global_model.rf_layers[layer_idx](X)
                R = labels - global_model.top_classifiers[layer_idx](X)

                if FF is None:
                    W = global_model.top_classifiers[layer_idx].W
                    FF = F.T @ F
                    WRF = W @ R.T @ F
                else:
                    W = global_model.top_classifiers[layer_idx].W
                    FF += F.T @ F
                    WRF += W @ R.T @ F

        return FF, WRF











