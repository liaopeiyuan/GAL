from helper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/node_attack.json')

    args = parser.parse_args()
    config = args.config_path
    with open(config) as f:
        d = json.load(f)

    print("Config:\n", json.dumps(d))

    def to_namespace(d, config):
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--config_path', type=str, default=config)
        for (k, v) in d.items():
            argparser.add_argument('--'+k, default=v)
        return argparser.parse_args()

    args = to_namespace(d, config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    args.train_ratings, args.test_ratings, args.users, args.movies = make_dataset_1M(True)
    dirname = os.path.join('./log', args.experiment, args.task, args.model)
    Path(dirname).mkdir(parents=True, exist_ok=True)

    def gen_str(L):
        final = ""
        for i in L:
            if len(final)>0: final += "_"+str(i)
            else :final=i
        return final

    logname = gen_str([args.task, args.model, args.gnn_type, args.gnn_layers,
                       args.embed_dim, args.batch_size,
                       args.lambda_reg, 
                       args.seed]) if args.task != 'NHop_Attack' else gen_str([args.task, args.model, args.gnn_type, args.gnn_layers,
                                                  args.embed_dim, args.batch_size,
                                                                         args.lambda_reg,
                                                                                                args.seed, args.hop])

    args.logname = logname
    args.logger = get_logger(logname+".txt", dirname, './config/')

    with open(os.path.join(dirname, config.split('/')[-1]), 'w') as outfile:
        json.dump(d, outfile)

    ''' Offset Movie ID's by # users because in TransD they share the same
    embedding Layer '''

    args.train_ratings['movie_id'] += int(np.max(args.users['user_id'])) + 1
    args.test_ratings['movie_id'] += int(np.max(args.users['user_id'])) + 1

    args.num_users = int(np.max(args.users['user_id'])) + 1
    args.num_movies = int(np.max(args.movies['movie_id'])) + 1
    args.num_ent = args.num_users + args.num_movies
    args.num_rel = 5
    users = np.asarray(list(set(args.users['user_id'])))
    np.random.shuffle(users)

    train_cutoff_row = int(np.round(len(users) * args.cutoff))
    args.cutoff_row = train_cutoff_row
    args.users_train = users[:train_cutoff_row]
    args.users_test = users[train_cutoff_row:]

    args.data_path = '../data/ml-1m'
    args.users.to_csv(args.data_path+"users.csv")
    args.movies.to_csv(args.data_path+"movies.csv")

    print("Task: ", args.task)
    print("Model: ", args.model)

    def fail_continuation(_): raise NotImplementedError

    if args.task == 'Node_Attack':
        if args.model == "Fixed_Embedding": from node_attack_fixed import run
        elif args.model == "Naive_GNN": 
            print("Ignoring lambda_reg field and setting it to 0.")
            args.lambda_reg=0
            from node_attack import run
        elif args.model == "GAL": 
            from node_attack import run
        elif args.model == "Noise": 
            from noise import run
        else: run = fail_continuation

    elif args.task == 'Neighbor_Attack':
        if args.model == "Fixed_Embedding": from neighbor_attack_fixed import run
        elif args.model == "Fixed_Embedding_NoTrain": from neighbor_attack_fixed import run
        elif args.model == "Naive_GNN": 
            print("Ignoring lambda_reg field and setting it to 0.")
            args.lambda_reg=0
            from neighbor_attack import run
        elif args.model == "GAL": 
            from neighbor_attack import run
        else: run = fail_continuation

    elif args.task == 'NHop_Attack':
        from n_hop import run

    else:
        run = fail_continuation

    run(args)
    return


if __name__ == '__main__':
    parse_args()
