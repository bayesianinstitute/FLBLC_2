from BCCommunicator import BCCommunicator
from FSCommunicator import FSCommunicator
from Model import Model
import torch
import os
from Requester import Requester
from Worker import Worker
from dotenv import load_dotenv
from FSCommunicator import FSCommunicator
import ipfshttpclient


# Main class to simulate the distributed application
class Application:

    def __init__(self, num_workers, num_rounds, ipfs_folder_hash, num_evil=0):
        self.client = ipfshttpclient.connect()
    
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.DEVICE = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.fspath =ipfs_folder_hash
        self.workers = []
        self.topk = num_workers
        self.worker_dict = {}
        self.num_evil = num_evil

    def run(self):
        load_dotenv()
        self.requester = Requester(os.getenv('REQUESTER_KEY'))
        self.requester.deploy_contract()
        self.requester.init_task(10000000000000000000, self.fspath, self.num_rounds)

        print("Task initialized")
       

        # in the beginning, all have the same model
        # the optimizer stays the same over all round
        # initialize all workers sequentially
        # in a real application, each device would run one worker class
        self.clusters = []  # Create a list to store clusters


        for i in range(self.num_workers):
            cluster_id = i % 2  # Assign cluster based on modulus
            worker = Worker(self.fspath, self.DEVICE, self.num_workers, i, 3, os.getenv('WORKER' + str(i+1) + '_KEY'), i < self.num_evil)
            self.worker_dict[i] = worker.account.address
            print("Account address:", self.worker_dict[i])
            
            worker.join_task(self.requester.get_contract_address(), cluster_id)  # Pass cluster_id to join_task
            
            # Add worker to the corresponding cluster
            if cluster_id >= len(self.clusters):
                self.clusters.append([])
            self.clusters[cluster_id].append(worker)
            print("Worker of the cluster id",self.clusters[cluster_id])
            

        self.requester.start_task()

        for round in range(self.num_rounds):
            print("Entered into Rounds", round)

            for cluster_id, cluster in enumerate(self.clusters):
                print("Entered into Cluster", cluster_id)
            
                for worker_idx, worker in enumerate(cluster):
                    print("Training of this Cluster", cluster_id, "Rounds", round, "Worker idx", worker_idx, "worker", worker)
                    worker.train(round, cluster_id,worker_idx)

                # for worker_idx, worker in enumerate(cluster):
                print("Average of this Cluster", cluster_id, "Rounds", round, "Worker idx", worker_idx, "worker", worker)
                average_cluster=worker.average(round, cluster_id,worker_idx)

                worker.sendModelipfs(average_cluster,round,cluster_id)



            


                # starting eval phase
                # for worker_idx, worker in enumerate(cluster):
            print("cluster:", cluster_id, "worker_idx:", worker_idx)
            avg_dicts, topK_dicts, unsorted_scores = worker.evaluate(round, cluster_id)
            unsorted_scores = [score[0].cpu().item() for score in unsorted_scores]
            unsorted_scores.insert(worker_idx, -1)
            unsorted_scores = (worker_idx, unsorted_scores)
                # print("unsorted scores:", unsorted_scores)
            self.requester.push_scores(unsorted_scores)
            worker.update_model(avg_dicts)

            overall_scores = self.requester.calc_overall_scores(
                self.requester.get_score_matrix(), self.num_workers)
            round_top_k = self.requester.compute_top_k(
                list(self.worker_dict.values()), overall_scores)
            
            # penalize = self.requester.find_bad_workers(
            #     list(self.worker_dict.values()), overall_scores)
            # print("penalize:", penalize)
            # self.requester.penalize_worker(penalize)
            # self.requester.refund_worker(list(self.worker_dict.values()))

            self.requester.submit_top_k(round_top_k)
            
            self.requester.distribute_rewards()
            print("Distributed rewards. Next round starting soon...")

            self.requester.next_round()