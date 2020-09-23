from helper import *
from data_loader import *

from model.models import *


class Runner(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format. 

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:        The dataloader for different data splits

        """

        if self.p.dataset == 'FB15k-237':
            et_path = './data/{}/Entity_Type_%s.txt'.format(self.p.dataset)

            train_file = open(et_path % 'train', 'r').readlines()
            valid_file = open(et_path % 'valid', 'r').readlines()
            test_file = open(et_path % 'test', 'r').readlines()
            train_data = parse_file(train_file)
            valid_data = parse_file(valid_file)
            test_data = parse_file(test_file)

            self.ent_to_idx, self.attr_to_idx = get_idx_dicts(
                train_data + valid_data + test_data)

            ''' Count attributes '''
            self.train_attr_count = count_attributes(
                train_data, self.attr_to_idx)
            self.valid_attr_count = count_attributes(
                valid_data, self.attr_to_idx)

            ''' Reindex Attribute Dictionary with 50 Most Common '''
            self.train_reindex_attr_idx = reindex_attributes(
                self.train_attr_count.most_common(50))

            attribute_mat = np.zeros((len(self.ent_to_idx), 50))
            self.attribute_mat = transform_data(train_data, self.ent_to_idx, self.attr_to_idx,
                                                self.train_reindex_attr_idx, attribute_mat)

            ent_set, rel_set = OrderedSet(), OrderedSet()
            for split in ['train', 'test', 'valid']:
                for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    ent_set.add(sub)
                    rel_set.add(rel)
                    ent_set.add(obj)

            self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
            self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
            self.rel2id.update({rel+'_reverse': idx+len(self.rel2id)
                                for idx, rel in enumerate(rel_set)})

            self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
            self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

            def gen_et_labels():
                list = []
                for i in self.id2ent.keys():
                    list.append(ent2label(i))
                list = torch.cat(list, dim=1)
                return list

            def ent2label(entity):
                try:
                    return torch.LongTensor(self.attribute_mat[self.ent_to_idx[self.id2ent[entity]], :]).unsqueeze(1)
                except:
                    return torch.zeros((50, 1)).long()

            self.et_labels = gen_et_labels().t()
            print(self.et_labels.sum())
            print(self.et_labels.size())

        elif self.p.dataset == 'WN18RR':
            ent_set, rel_set = OrderedSet(), OrderedSet()
            for split in ['train', 'test', 'valid']:
                for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    ent_set.add(sub)
                    rel_set.add(rel)
                    ent_set.add(obj)

            self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
            self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
            self.rel2id.update({rel+'_reverse': idx+len(self.rel2id)
                                for idx, rel in enumerate(rel_set)})

            self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
            self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

            tags = pd.read_csv(
                './data/WN18RR/WN18RR_tags_mlj13.csv', dtype={'id': object})

            mapping = {'JJ': 0, 'NN': 1, 'RB': 2, 'VB': 3}
            def sense_map(x): return min(x, 20)-1

            labels = []
            for row in tqdm(tags.iterrows()):
                val = row[1]
                try:
                    labels.append(
                        [self.ent2id[val['id']], mapping[val['POS_tag']], sense_map(val['sense'])])
                except:
                    print(val['id'])

            labels.sort(key=lambda x: x[0])

            prev = None
            for line in labels:
                if (prev == None):
                    prev = line[0]
                else:
                    assert(line[0]-prev == 1)
                    prev = line[0]

            labels = torch.tensor(np.array(labels)[:, 1:]).long()
            print(labels.size())

            self.et_labels = labels

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * \
            self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append(
                {'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj),        'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        print(self.triples.keys())

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train':        get_data_loader(TrainDataset, 'train',         self.p.batch_size),
            'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
            'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
            'test_head':       get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
            'test_tail':       get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

        print("edge index: ",self.edge_index.size())
        print(self.p.num_ent)
        print(self.p.num_rel) 

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type). to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p = params

        def gen_str(L):
            final = ""
            for i in L:
                if len(final)>0: final += "_"+str(i)
                else :final=i
            return final

        logname = gen_str([params.dataset,params.batch_size,
                           params.score_func,
                           params.seed,params.lambda_reg,
                           params.gcn_layer,"gal.txt"])

        self.logger = get_logger(
            logname, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.lambda_reg = self.p.lambda_reg
        self.load_data()
        self.model = self.add_model(self.p.model, self.p.score_func)

        model_name = '{}_{}'.format(args.model, args.score_func)

        conv_params = [{'params': self.model.conv1.w_loop},
                       {'params': self.model.conv1.w_in},
                       {'params': self.model.conv1.w_out},
                       {'params': self.model.conv1.w_rel},
                       {'params': self.model.conv1.loop_rel},
                       {'params': self.model.conv1.bn.weight},
                       {'params': self.model.conv1.bn.bias}]

        if self.p.gcn_layer == 2:
            conv_params = [{'params': self.model.conv1.w_loop},
                       {'params': self.model.conv1.w_in},
                       {'params': self.model.conv1.w_out},
                       {'params': self.model.conv1.w_rel},
                       {'params': self.model.conv1.loop_rel},
                       {'params': self.model.conv1.bn.weight},
                       {'params': self.model.conv1.bn.bias},
                       {'params': self.model.conv2.w_loop},
                    {'params': self.model.conv2.w_in},
                    {'params': self.model.conv2.w_out},
                    {'params': self.model.conv2.w_rel},
                    {'params': self.model.conv2.loop_rel},
                    {'params': self.model.conv2.bn.weight},
                    {'params': self.model.conv2.bn.bias},]

        init_params = [{'params': self.model.init_embed},
                       {'params': self.model.init_rel},
                       {'params': self.model.bias}, ]

        if model_name.lower() == 'compgcn_conve':
            self.optimizer = self.add_optimizer([
                {'params': self.model.bn0.weight},
                {'params': self.model.bn1.weight},
                {'params': self.model.bn2.weight},
                {'params': self.model.bn0.bias},
                {'params': self.model.bn1.bias},
                {'params': self.model.bn2.bias},
                {'params': self.model.m_conv1.weight},

                {'params': self.model.fc.weight},
                {'params': self.model.fc.bias},
                {'params': self.model.init_embed},
                       {'params': self.model.init_rel},
                       {'params': self.model.bias}, {'params': self.model.conv1.w_loop},
                       {'params': self.model.conv1.w_in},
                       {'params': self.model.conv1.w_out},
                       {'params': self.model.conv1.w_rel},
                       {'params': self.model.conv1.loop_rel},
                       {'params': self.model.conv1.bn.weight},
                       {'params': self.model.conv1.bn.bias}])
        else:
            self.optimizer = self.add_optimizer([{'params': self.model.init_embed},
                       {'params': self.model.init_rel},
                       {'params': self.model.bias},{'params': self.model.conv1.w_loop},
                       {'params': self.model.conv1.w_in},
                       {'params': self.model.conv1.w_out},
                       {'params': self.model.conv1.w_rel},
                       {'params': self.model.conv1.loop_rel},
                       {'params': self.model.conv1.bn.weight},
                       {'params': self.model.conv1.bn.bias}])

        self.attr_optimizer = self.add_optimizer([
            {'params': self.model.attr_predictor[0].weight},
            {'params': self.model.attr_predictor[0].bias},
            {'params': self.model.attr_predictor[2].weight},
            {'params': self.model.attr_predictor[2].bias},
            {'params': self.model.attr_predictor[4].weight},
            {'params': self.model.attr_predictor[4].bias},
            {'params': self.model.init_embed},
                       {'params': self.model.init_rel},
                       {'params': self.model.bias},
                       {'params': self.model.conv1.w_loop},
                       {'params': self.model.conv1.w_in},
                       {'params': self.model.conv1.w_out},
                       {'params': self.model.conv1.w_rel},
                       {'params': self.model.conv1.loop_rel},
                       {'params': self.model.conv1.bn.weight},
                       {'params': self.model.conv1.bn.bias}],mult=self.lambda_reg)

        self.attacker_optimizer = self.add_optimizer([
            {'params': self.model.attacker[0].weight},
            {'params': self.model.attacker[0].bias},
            {'params': self.model.attacker[2].weight},
            {'params': self.model.attacker[2].bias},
            {'params': self.model.attacker[4].weight},
            {'params': self.model.attacker[4].bias},
        ])

        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])
        print("=============")
        for param_group in self.attr_optimizer.param_groups:
            print(param_group['lr'])

        if self.p.dataset == 'WN18RR':
            self.attr2_optimizer = self.add_optimizer([
                {'params': self.model.attr2_predictor[0].weight},
                {'params': self.model.attr2_predictor[0].bias},
                {'params': self.model.attr2_predictor[2].weight},
                {'params': self.model.attr2_predictor[2].bias},
                {'params': self.model.attr2_predictor[4].weight},
                {'params': self.model.attr2_predictor[4].bias},
               {'params': self.model.init_embed},
                       {'params': self.model.init_rel},
                       {'params': self.model.bias},{'params': self.model.conv1.w_loop},
                       {'params': self.model.conv1.w_in},
                       {'params': self.model.conv1.w_out},
                       {'params': self.model.conv1.w_rel},
                       {'params': self.model.conv1.loop_rel},
                       {'params': self.model.conv1.bn.weight},
                       {'params': self.model.conv1.bn.bias}],mult=self.lambda_reg)

            self.attacker2_optimizer = self.add_optimizer([
                {'params': self.model.attacker2[0].weight},
                {'params': self.model.attacker2[0].bias},
                {'params': self.model.attacker2[2].weight},
                {'params': self.model.attacker2[2].bias},
                {'params': self.model.attacker2[4].weight},
                {'params': self.model.attacker2[4].bias},
            ])

    def add_model(self, model, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """
        model_name = '{}_{}'.format(model, score_func)

        if model_name.lower() == 'compgcn_transe':
            model = CompGCN_TransE(
                self.edge_index, self.edge_type, params=self.p, et_labels=self.et_labels, dataset=self.p.dataset)
        elif model_name.lower() == 'compgcn_distmult':
            model = CompGCN_DistMult(
                self.edge_index, self.edge_type, params=self.p, et_labels=self.et_labels, dataset=self.p.dataset)
        elif model_name.lower() == 'compgcn_conve':
            model = CompGCN_ConvE(
                self.edge_index, self.edge_type, params=self.p, et_labels=self.et_labels, dataset=self.p.dataset)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, parameters, mult=None):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        if mult is None:
            return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
        else:
            return torch.optim.Adam(parameters, lr=self.p.lr*mult, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch:         the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        Head, Relation, Tails, labels
        """
        if split == 'train':
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict'    : self.model.state_dict(),
            'best_val'    : self.best_val,
            'best_epoch'    : self.best_epoch,
            'optimizer'    : self.optimizer.state_dict(),
            'args'        : vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:            The evaluation results containing the following:
                                        results['mr']:             Average of ranks_left and ranks_right
                                        results['mrr']:         Mean Reciprocal Rank
                                        results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        if (self.p.dataset == "WN18RR"):

            left_results = self.predict(split=split, mode='tail_batch')
            right_results = self.predict(split=split, mode='head_batch')
            attr_results = self.predict_attr(split=split, mode='head_batch')
            attr_results2 = self.predict_attr2(split=split, mode='head_batch')
            attack_results = self.predict_attack(
                split=split, mode='head_batch')
            attack_results2 = self.predict_attack2(
                split=split, mode='head_batch')

            micro = attr_results['micro']
            macro = attr_results['macro']

            micro2 = attr_results2['micro']
            macro2 = attr_results2['macro']

            amicro = attack_results['micro']
            amacro = attack_results['macro']

            amicro2 = attack_results2['micro']
            amacro2 = attack_results2['macro']

            self.logger.info('[Epoch {} {}]: Attr: Loss : {:.5}, Macro-F1 : {:.5}, Micro-F1 : {:.5}'.format(
                epoch, split, attr_results['avg_loss'], macro, micro))
            self.logger.info('[Epoch {} {}]: Attr2: Loss : {:.5}, Macro-F1 : {:.5}, Micro-F1 : {:.5}'.format(
                epoch, split, attr_results2['avg_loss'], macro2, micro2))

            self.logger.info('[Epoch {} {}]: Attack: Loss : {:.5}, Macro-F1 : {:.5}, Micro-F1 : {:.5}'.format(
                epoch, split, attack_results['avg_loss'], amacro, amicro))
            self.logger.info('[Epoch {} {}]: Attack2: Loss : {:.5}, Macro-F1 : {:.5}, Micro-F1 : {:.5}'.format(
                epoch, split, attack_results2['avg_loss'], amacro2, amicro2))

            results = get_combined_results(left_results, right_results)
            self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(
                epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
            self.logger.info('[Epoch {} {}]: MR: {:.5}, Hits@10 : {:.5}'.format(
                epoch, split, results['mr'], results['hits@10']))

        elif (self.p.dataset == "FB15k-237"):

            left_results = self.predict(split=split, mode='tail_batch')
            right_results = self.predict(split=split, mode='head_batch')
            attr_results = self.predict_attr(split=split, mode='head_batch')
            attack_results = self.predict_attack(
                split=split, mode='head_batch')

            macro_attack = f1_score(
                attack_results['truths'], (attack_results['outs'] > 0.5).astype(int), average='macro')
            auc_attack = roc_auc_score(
                attack_results['truths'], attack_results['outs'])

            macro = f1_score(attr_results['truths'], (attr_results['outs'] > 0.5).astype(
                int), average='macro')
            auc = roc_auc_score(attr_results['truths'], attr_results['outs'])
            self.logger.info('[Epoch {} {}]: Attr: Loss : {:.5}, AUC : {:.5}, Macro-F1 : {:.5}'.format(
                epoch, split, attr_results['avg_loss'], auc, macro))
            self.logger.info('[Epoch {} {}]: Attack: Loss : {:.5}, AUC : {:.5}, Macro-F1 : {:.5}'.format(
                epoch, split, attack_results['avg_loss'], auc_attack, macro_attack))

            results = get_combined_results(left_results, right_results)
            self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(
                epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
            self.logger.info('[Epoch {} {}]: MR: {:.5}, Hits@10 : {:.5}'.format(
                epoch, split, results['mr'], results['hits@10']))

        return results

    def predict_attr2(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string)     If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):        Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:            The evaluation results containing the following:
                results['mr']:             Average of ranks_left and ranks_right
                results['mrr']:         Mean Reciprocal Rank
                results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(
                self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            losses_attr = []
            outs = []
            truths = []
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                loss, out, truth = self.model.forward_attr2(sub, rel)
                pred = self.model.forward(sub, rel)

                losses_attr.append(loss.item())

                outs.append(out)

                truths.append(truth)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(
                        split.title(), mode.title(), step, self.p.name))

        outs = torch.cat(outs, dim=0).detach().cpu()
        truths = torch.cat(truths, dim=0).detach().cpu().numpy()
        results['avg_loss'] = np.mean(losses_attr)
        results['outs'] = outs.numpy()
        results['truths'] = truths
        micro = f1_score(truths, (torch.argmax(
            outs, dim=1)).numpy(), average='micro')
        results['micro'] = micro
        macro = f1_score(truths, (torch.argmax(
            outs, dim=1)).numpy(), average='macro')
        results['macro'] = macro
        return results

    def predict_attr(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string)     If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):        Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:            The evaluation results containing the following:
                                        results['mr']:             Average of ranks_left and ranks_right
                                        results['mrr']:         Mean Reciprocal Rank
                                        results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(
                self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            losses_attr = []
            outs = []
            truths = []
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                loss, out, truth = self.model.forward_attr(sub, rel)

                losses_attr.append(loss.item())

                outs.append(out)

                truths.append(truth)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(
                        split.title(), mode.title(), step, self.p.name))

        truths = torch.cat(truths, dim=0).detach().cpu().numpy()
        results['avg_loss'] = np.mean(losses_attr)
        results['outs'] = torch.cat(outs, dim=0).detach().cpu().numpy()
        results['truths'] = truths

        if (self.p.dataset == 'WN18RR'):
            outs=torch.cat(outs, dim=0).detach().cpu()
            micro = f1_score(truths, (torch.argmax(
                outs, dim=1)).numpy(), average='micro')
            results['micro'] = micro
            macro = f1_score(truths, (torch.argmax(
                outs, dim=1)).numpy(), average='macro')
            results['macro'] = macro
        return results

    def predict_attack(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string)     If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):        Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:            The evaluation results containing the following:
                                        results['mr']:             Average of ranks_left and ranks_right
                                        results['mrr']:         Mean Reciprocal Rank
                                        results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(
                self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            losses_attr = []
            outs = []
            truths = []
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                loss, out, truth = self.model.forward_attack(sub, rel)

                losses_attr.append(loss.item())

                outs.append(out)

                truths.append(truth)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(
                        split.title(), mode.title(), step, self.p.name))

        truths = torch.cat(truths, dim=0).detach().cpu().numpy()
        results['avg_loss'] = np.mean(losses_attr)
        results['outs'] = torch.cat(outs, dim=0).detach().cpu().numpy()
        results['truths'] = truths

        if (self.p.dataset == 'WN18RR'):
            outs=torch.cat(outs, dim=0).detach().cpu()
            micro = f1_score(truths, (torch.argmax(
                outs, dim=1)).numpy(), average='micro')
            results['micro'] = micro
            macro = f1_score(truths, (torch.argmax(
                outs, dim=1)).numpy(), average='macro')
            results['macro'] = macro
        return results

    def predict_attack2(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string)     If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):        Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:            The evaluation results containing the following:
                results['mr']:             Average of ranks_left and ranks_right
                results['mrr']:         Mean Reciprocal Rank
                results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(
                self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            losses_attr = []
            outs = []
            truths = []
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                loss, out, truth = self.model.forward_attack2(sub, rel)
                pred = self.model.forward(sub, rel)

                losses_attr.append(loss.item())

                outs.append(out)

                truths.append(truth)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(
                        split.title(), mode.title(), step, self.p.name))

        outs = torch.cat(outs, dim=0).detach().cpu()
        truths = torch.cat(truths, dim=0).detach().cpu().numpy()
        results['avg_loss'] = np.mean(losses_attr)
        results['outs'] = outs.numpy()
        results['truths'] = truths
        micro = f1_score(truths, (torch.argmax(
            outs, dim=1)).numpy(), average='micro')
        results['micro'] = micro
        macro = f1_score(truths, (torch.argmax(
            outs, dim=1)).numpy(), average='macro')
        results['macro'] = macro
        return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string)     If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):        Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:            The evaluation results containing the following:
                                        results['mr']:             Average of ranks_left and ranks_right
                                        results['mrr']:         Mean Reciprocal Rank
                                        results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(
                self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred = self.model.forward(sub, rel)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(
                    label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1,
                                                        descending=True), dim=1, descending=False)[b_range, obj]

                ranks = ranks.float()
                results['count'] = torch.numel(
                    ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(
                    ranks).item() + results.get('mr',    0.0)
                results['mrr'] = torch.sum(
                    1.0/ranks).item() + results.get('mrr',   0.0)
                for k in range(10):
                    results['hits@{}'.format(
                        k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(
                        split.title(), mode.title(), step, self.p.name))

        return results

    def run_epoch_attr(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        losses_attr = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):

            sub, rel, obj, label = self.read_batch(batch, 'train')

            self.optimizer.zero_grad()

            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if (step % 1 == 0 and epoch > 35):
                for _ in range(1):
                    self.attr_optimizer.zero_grad()
                    loss_attr, _, _ = self.model.forward_attr(sub, rel, attr=0)
                    loss_attr.backward()
                    self.attr_optimizer.step()
                    losses_attr.append(loss_attr.item())
            else:
                losses_attr.append(0)
                
            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train :{:.5}, Attr: {:.5}, Val MRR:{:.5}\t{}'.format(
                    epoch, step, np.mean(losses), np.mean(losses_attr), self.best_val_mrr, self.p.name))

        loss = np.mean(losses_attr)
        self.logger.info(
            '[Epoch:{}]:  Train Attr Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def run_epoch_attr2(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        losses_attr = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):

            sub, rel, obj, label = self.read_batch(batch, 'train')

            self.optimizer.zero_grad()
            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if (step % 1 == 0 and epoch > 35):
                for _ in range(1):
                    self.attr2_optimizer.zero_grad()
                    loss_attr, _, _ = self.model.forward_attr2(sub, rel, attr=0)
                    loss_attr.backward()
                    self.attr2_optimizer.step()
                    losses_attr.append(loss_attr.item())
            else:
                losses_attr.append(0)
                
            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train :{:.5}, Attr: {:.5}, Val MRR:{:.5}\t{}'.format(
                    epoch, step, np.mean(losses), np.mean(losses_attr), self.best_val_mrr, self.p.name))

        loss = np.mean(losses_attr)
        self.logger.info(
            '[Epoch:{}]:  Train Attr Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def run_epoch_attack(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        losses_attr = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):

            self.attacker_optimizer.zero_grad()

            sub, rel, obj, label = self.read_batch(batch, 'train')

            loss_attr, _, _ = self.model.forward_attack(sub, rel, attr=0)

            loss_attr.backward()
            self.attacker_optimizer.step()
            losses_attr.append(loss_attr.item())

            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)

            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train :{:.5}, Attack: {:.5}, Val MRR:{:.5}\t{}'.format(
                    epoch, step, np.mean(losses), np.mean(losses_attr), self.best_val_mrr, self.p.name))

        loss = np.mean(losses_attr)
        self.logger.info(
            '[Epoch:{}]:  Training Attr Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def run_epoch_attack2(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        losses_attr = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):

            self.attacker2_optimizer.zero_grad()

            sub, rel, obj, label = self.read_batch(batch, 'train')

            loss_attr, _, _ = self.model.forward_attack2(sub, rel, attr=0)

            loss_attr.backward()
            self.attacker2_optimizer.step()
            losses_attr.append(loss_attr.item())

            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)

            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train :{:.5}, Attack: {:.5}, Val MRR:{:.5}\t{}'.format(
                    epoch, step, np.mean(losses), np.mean(losses_attr), self.best_val_mrr, self.p.name))

        loss = np.mean(losses_attr)
        self.logger.info(
            '[Epoch:{}]:  Training Attr Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def run_epoch(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        losses_attr = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):

            sub, rel, obj, label = self.read_batch(batch, 'train')

            loss_attr, _, _ = self.model.forward_attr(sub, rel, attr=0)

            self.optimizer.zero_grad()

            losses_attr.append(loss_attr.item())

            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train :{:.5}, Attr: {:.5}, Val MRR:{:.5}\t{}'.format(
                    epoch, step, np.mean(losses), np.mean(losses_attr), self.best_val_mrr, self.p.name))

        loss = np.mean(losses)
        self.logger.info(
            '[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        for epoch in range(self.p.max_epochs):
            train_loss = self.run_epoch_attr(epoch, val_mrr)
            val_results = self.evaluate('valid', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)

            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(
                epoch, train_loss, self.best_val_mrr))

        for epoch in range(30):
            train_loss = self.run_epoch_attack(epoch, val_mrr)
            val_results = self.evaluate('test', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)

            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(
                epoch, train_loss, self.best_val_mrr))

        train_loss = self.run_epoch_attr(epoch, val_mrr)
        self.logger.info('Loading best model, Evaluating on Test data')
        test_results = self.evaluate('test', epoch)

    def fit2(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        for epoch in range(self.p.max_epochs):
            train_loss = self.run_epoch_attr2(epoch, val_mrr)
            val_results = self.evaluate('valid', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)

            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(
                epoch, train_loss, self.best_val_mrr))

        for epoch in range(30):
            train_loss = self.run_epoch_attack2(epoch, val_mrr)
            val_results = self.evaluate('test', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)

            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(
                epoch, train_loss, self.best_val_mrr))

        self.logger.info('Loading best model, Evaluating on Test data')
        test_results = self.evaluate('test', epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',        default='testrun',
                        help='Set run name for saving/restoring models')
    parser.add_argument('-data',        dest='dataset',         default='FB15k-237',
                        help='Dataset to use, options: FB15k-237, WN18RR')
    parser.add_argument('-model',        dest='model',
                        default='compgcn',        help='Model Name')
    parser.add_argument('-score_func',    dest='score_func',
                        default='conve',        help='Score Function for Link prediction')
    parser.add_argument('-opn',             dest='opn',             default='corr',
                        help='Composition Operation to be used in CompGCN')

    parser.add_argument('-batch',           dest='batch_size',
                        default=128,    type=int,       help='Batch size')
    parser.add_argument('-gamma',        type=float,
                        default=40.0,            help='Margin')
    parser.add_argument('-gpu',        type=str,               default='0',
                        help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',        dest='max_epochs',     type=int,
                        default=1,      help='Number of epochs')
    parser.add_argument('-l2',        type=float,
                        default=0.0,            help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',        type=float,
                        default=0.001,            help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth',      dest='lbl_smooth',
                        type=float,     default=0.1,    help='Label Smoothing')
    parser.add_argument('-num_workers',    type=int,               default=10,
                        help='Number of processes to construct batches')
    parser.add_argument('-seed',            dest='seed',
                        default=114514,  type=int,         help='Seed for randomization')

    parser.add_argument('-restore',         dest='restore',         action='store_true',
                        help='Restore from the previously saved model')
    parser.add_argument('-bias',            dest='bias',
                        action='store_true',            help='Whether to use bias in the model')

    parser.add_argument('-num_bases',    dest='num_bases',     default=-1,
                        type=int,     help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',    dest='init_dim',    default=100,    type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',          dest='gcn_dim',     default=200,
                        type=int,     help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',    dest='embed_dim',     default=None,   type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',    dest='gcn_layer',     default=1,
                        type=int,     help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',    dest='dropout',     default=0.1,
                        type=float,    help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',      dest='hid_drop',
                        default=0.3,      type=float,    help='Dropout after GCN')
    parser.add_argument('-lambda',      dest='lambda_reg',
                        default=1e4,      type=float,    help='Lambda term for gradient descent-ascent')


    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2',      dest='hid_drop2',
                        default=0.3,      type=float,    help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop',     dest='feat_drop',
                        default=0.3,      type=float,    help='ConvE: Feature Dropout')
    parser.add_argument('-k_w',          dest='k_w',         default=10,
                        type=int,     help='ConvE: k_w')
    parser.add_argument('-k_h',          dest='k_h',         default=20,
                        type=int,     help='ConvE: k_h')
    parser.add_argument('-num_filt',      dest='num_filt',     default=200,
                        type=int,     help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz',        dest='ker_sz',         default=7,
                        type=int,     help='ConvE: Kernel size to use')

    parser.add_argument('-logdir',          dest='log_dir',
                        default='./log/',               help='Log directory')
    parser.add_argument('-config',          dest='config_dir',
                        default='./config/',            help='Config directory')
    parser.add_argument('-experiment',          dest='experiment',      default='ablation',            help='Type of experiment')
    args = parser.parse_args()

    if not args.restore:
        args.name = args.name + '_' + \
            time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    model = Runner(args)
    model.fit()

    def gen_str(L):
        final = ""
        for i in L:
            if len(final)>0: final += "_"+str(i)
            else :final=i
        return final

    if args.dataset == 'WN18RR':
        del(model)
        gc.collect()

        model2 = Runner(args)
        model2.fit2()

        logname = gen_str( [args.experiment, args.dataset,args.batch_size,
                        args.score_func,
                        args.seed,
                        args.gcn_layer,args.lambda_reg,"gal_attr2.pth"])

        path = os.path.join("./checkpoint",logname)

        torch.save(model2.model.state_dict(), path)

    logname = gen_str( [args.experiment, args.dataset,args.batch_size,
                        args.score_func,
                        args.seed,
                        args.gcn_layer,args.lambda_reg,"gal_attr1.pth"])

    path = os.path.join("./checkpoint",logname)

    torch.save(model.model.state_dict(), path)
