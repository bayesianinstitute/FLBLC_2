import json
from web3 import Web3, HTTPProvider

import torch.optim as optim
from BCCommunicator import BCCommunicator
from FSCommunicator import FSCommunicator
from Model import Model
import torch
import os 
import glob

# this simulates one worker. Usually each device has one of these

class Worker:
    truffle_file = json.load(open('./build/contracts/FLTask.json'))
    
    def __init__(self, ipfs_path, device, num_workers, idx, topk, key, is_evil):
        self.bcc = BCCommunicator()
        self.fsc = FSCommunicator(ipfs_path, device)
        

        model, opt = self.fsc.fetch_initial_model()
        self.is_evil = is_evil
        
        # This is incredibly hacky and should definitely not be done like this
        # we use pythons reflection ability to create the correct optimizer
        # it is not clear if that works for a general optimizer or only for opt.SGD
        class_ = getattr(optim, opt['name'])
        copy = dict(opt['state_dict']['param_groups'][0])
        try:
            del copy['params']
        except:
            pass
        opt = class_(model.parameters(), **(copy))
        # print("opt",opt)
        self.model = Model(num_workers, idx, model, opt, device, topk, is_evil)
        # print("model value",self.model)
        self.idx = idx
        self.num_workers = num_workers

        self.key = key
        # init web3.py instance
        self.w3 = Web3(HTTPProvider("http://localhost:7545"))
        if(self.w3.isConnected()):
            print("Worker initialization: connected to blockchain")

        self.account = self.w3.eth.account.privateKeyToAccount(key)
        self.contract = self.w3.eth.contract(bytecode=self.truffle_file['bytecode'], abi=self.truffle_file['abi'])
        
        
    def train(self, round, clusterid,worker_id):
    # train

        cur_state_dict = self.model.train()

        # Store cur_state_dict based on clusterid
        save_path = f"cluster_model/model_workerid_{worker_id}_cluster_id_{clusterid}_round_{round}.pt"
        torch.save(cur_state_dict, save_path)

        # print("CUR",cur_state_dict)
       
        # push to file system
       
        #changes
        # self.fsc.push_model(cur_state_dict, self.idx, round,clusterid,self.num_workers)  
        


        # print("Model push update:",self.fsc.push_model(cur_state_dict,self.idx,round))
    
    # internal Averaging function
    def average(self, round, clusterid, worker_id):
        model_paths = []
        print(f"worker_id : {worker_id} cluster_id : {clusterid} round : {round}")
        folder_path='cluster_model'
        
        # Find all model paths for the given clusterid
        model_paths = glob.glob(f"{folder_path}/*.pt")
        models = []
        
        for path in model_paths:
            model = torch.load(path)
            models.append(model)
        
        # return models
        print("Model Length in list: ", len(models))
        
        # Calculate average weight
        average_state_dict = {}
        for key in models[0].keys():
            average_state_dict[key] = torch.zeros_like(models[0][key])
            for model in models:
                average_state_dict[key] += model[key]
            average_state_dict[key] /= len(models)

        print(f"Average Done!! For Cluster : {clusterid}")
        # Remove all files in folder
        file_list = os.listdir(folder_path)
    
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return average_state_dict
        # model.eval()
    
    def sendModelipfs(self,cur_state_dict,round,clusterid):
        self.fsc.push_model(cur_state_dict, self.idx, round,clusterid,self.num_workers)  

    def evaluate(self, round,cluster_id):
        print("Evaluating")
        # retrieve all models of the other workers
        state_dicts = self.fsc.fetch_evaluation_models(self.idx, round, self.num_workers,cluster_id)
        
        ranks, topk_dicts, unsorted_scores = self.model.eval(state_dicts)
        
        # add our own model for the averaging
        topk_dicts.append(self.model.model.state_dict())
        
        # @TODO add blockchain functionality with sending the ranks to BC here
        
        return self.model.average(topk_dicts), topk_dicts, unsorted_scores
        
    
    
    def update_model(self, avg_dicts):
        # here we update the model with the averaged dicts
        self.model.adapt_current_model(avg_dicts)

    def join_task(self, contract_address, cluster_id):
        self.contract_address = contract_address
        self.contract_instance = self.w3.eth.contract(abi=self.truffle_file['abi'], address=contract_address)
        deposit = 5000000000000000000  # 5 ethers (in wei)
        tx = self.contract_instance.functions.joinTask(cluster_id).buildTransaction({
            "gasPrice": self.w3.eth.gas_price,
            "chainId": 1337,
            "from": self.account.address,
            "value": deposit,
            'nonce': self.w3.eth.getTransactionCount(self.account.address)
        })
        # Get tx receipt to get contract address
        signed_tx = self.w3.eth.account.signTransaction(tx, self.key)
        tx_hash = self.w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.getTransactionReceipt(tx_hash)


    def get_model_uri(self):
        return self.contract_instance.functions.getModelURI().call()

    def get_round_number(self):
        return self.contract_instance.functions.getRound().call()

       
