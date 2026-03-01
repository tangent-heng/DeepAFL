import os
import time

import torch

from .deep_afl import DeepAFL


class FederatedServer:
    def __init__(self, backbone, train_set, test_set, args, device):
        self.backbone = backbone
        self.test_set = test_set
        self.args = args
        self.device = device
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

        # Initialize global model
        self.global_model = DeepAFL(
            in_dim=args.feat_size,
            out_dim=args.num_classes,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            randfeat_xt_dim=args.fe,
            l2_reg=args.reg,
            l2_sand=args.reg_sand,
            activation=args.activation,
        )

        self.corr_X0 = None
        self.corr_Xt = None
        self.corr_Y0 = None
        self.corr_Yt = None


        self.A = None
        self.B = None


    def get_global_model(self):
        """
        Returns the global model.
        """
        return self.global_model

    def aggregate_correlation(self, corr_Xt, corr_Yt):
        """
        Aggregate correlation matrices from clients.
        """
        if self.corr_Xt is None:
            self.corr_Xt = corr_Xt
            self.corr_Yt = corr_Yt
        else:
            self.corr_Xt += corr_Xt
            self.corr_Yt += corr_Yt


    def update_global_model(self, n, aggregate_type="top_classifiers"):
        """
        Update the global model with aggregated correlation matrices.
        """
        if aggregate_type == "top_classifiers":
            self.global_model.update_top_classifiers(
                self.corr_Xt, self.corr_Yt, n
            )
            self.corr_Xt = None
            self.corr_Yt = None
        elif aggregate_type == "sandwiched_layers":
            W = self.global_model.top_classifiers[n].W
            self.global_model.update_sandwiched_layers(self.A, self.B, W, n)
            self.A = None
            self.B = None
        else:
            raise ValueError("Unknown aggregation type: {}".format(aggregate_type))

    def aggregate_sandwich(self, A, B):
        """
        Aggregate sandwich matrices from clients.
        """
        if self.A is None:
            self.A = A
            self.B = B
        else:
            self.A += A
            self.B += B

    def evaluate(self):
        """
        Evaluate the global model on the test set.
        """
        self.global_model.eval()
        total_correct = {}
        total_samples = {}


        start_time = time.time()

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Extract features and get predictions
                features = self.backbone(images)
                outputs = self.global_model(features)

                # Calculate correct predictions
                for key in outputs.keys():
                    if key not in total_correct:
                        total_correct[key] = 0
                        total_samples[key] = 0

                    total_correct[key] += sum(outputs[key].argmax(dim=1) == targets.argmax(dim=1))
                    total_samples[key] += targets.size(0)

        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")

                # total_correct += sum(outputs.argmax(dim=1) == targets.argmax(dim=1))
                # total_samples += targets.size(0)

        # accuracy = total_correct / total_samples
        accuracy={}
        for key in total_correct.keys():
            accuracy[key] = total_correct[key] / total_samples[key]
            print(f"Accuracy for {key}: {accuracy[key]:.4f}")
        # print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_(self):
        """
        Evaluate the global model on the train set.
        """
        self.global_model.eval()
        total_correct = {}
        total_samples = {}

        with torch.no_grad():
            for images, targets in self.train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Extract features and get predictions
                features = self.backbone(images)
                outputs = self.global_model(features)

                # Calculate correct predictions
                for key in outputs.keys():
                    if key not in total_correct:
                        total_correct[key] = 0
                        total_samples[key] = 0

                    total_correct[key] += sum(outputs[key].argmax(dim=1) == targets.argmax(dim=1))
                    total_samples[key] += targets.size(0)

                # total_correct += sum(outputs.argmax(dim=1) == targets.argmax(dim=1))
                # total_samples += targets.size(0)

        # accuracy = total_correct / total_samples
        accuracy={}
        for key in total_correct.keys():
            accuracy[key] = total_correct[key] / total_samples[key]
            print(f"Accuracy for {key}: {accuracy[key]:.4f}")
        # print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy





















