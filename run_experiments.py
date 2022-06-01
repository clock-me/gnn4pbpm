from pipeline import run_pipeline
from pipeline import Config
import optimal_hyperparams
from argparse import ArgumentError, ArgumentParser
import wandb
import gc

def make_experiment_name(model, task, dataset_name, k):
    result = f'{model}__{task}__{dataset_name}_time_{k}'
    return result

def run_sequential_baselines(
    base_config,
    group_name,
):
    task = base_config.task
    data_module = None
    for time_type in ['absolute']:
        for k in [None, 0, 1, 20]:
            for model in [
                'gru',
                'transformer',
            ]:
                model_config = optimal_hyperparams.OPT_HYPERPARAMS[model]  
                base_config.model = model
                base_config.task = task
                base_config.model_config = model_config
                base_config.run_name = make_experiment_name(model, task, base_config.dataset, k) 
                base_config.group = group_name
                base_config.time_type = time_type   
                if model == 'gru' or model == 'transformer':
                    base_config.batch_size = 128
                else:
                    base_config.batch_size = 32
                base_config.model_config['time2vec_k'] = k
                data_module = run_pipeline(base_config, data_module)
                wandb.join()
                wandb.finish()
                gc.collect()
            gc.collect()
        gc.collect()

def run_gat_only_with_ablations(
    base_config,
    group_name,
):
    data_module = None
    for graph_type in ['bpmn']:     
        for (use_type_features, use_time_features, use_freq_features) in [
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        ]:
            task = base_config.task
            time_type = 'relative'
            model = 'gat'
            model_config = optimal_hyperparams.OPT_HYPERPARAMS[model]  
            base_config.model = model
            base_config.task = task
            base_config.model_config = model_config
            base_config.model_config['add_time_features'] = use_time_features
            base_config.model_config['add_freq_features'] = use_freq_features
            base_config.model_config['add_type_features'] = use_type_features
            base_config.graph_builder_config['graph_type'] = graph_type
            base_config.run_name = make_experiment_name(model, task, base_config.dataset, None) 
            base_config.group = group_name
            base_config.time_type = time_type   
            base_config.batch_size = 256
            # base_config.model_config['time2vec_k'] = k
            data_module = run_pipeline(base_config, data_module)
            wandb.join()
            wandb.finish()
            gc.collect()
            
def run_gcn_only(
    base_config,
    group_name,
):
    data_module = None
    for graph_type in ['bpmn']:     
        for use_freq_features in [
            True,
        ]:
            task = base_config.task
            time_type = 'relative'
            model = 'gcn'
            model_config = optimal_hyperparams.OPT_HYPERPARAMS[model]  
            base_config.model = model
            base_config.task = task
            base_config.model_config = model_config
            
            base_config.model_config['add_freq_features'] = use_freq_features

            base_config.graph_builder_config['graph_type'] = graph_type
            base_config.run_name = make_experiment_name(model, task, base_config.dataset, None) 
            base_config.group = group_name
            base_config.time_type = time_type   
            base_config.batch_size = 128
            # base_config.model_config['time2vec_k'] = k
            data_module = run_pipeline(base_config, data_module)
            wandb.join()
            wandb.finish()
            gc.collect()

def run_ggnn_only(
    base_config,
    group_name,
):
    data_module = None
    for graph_type in ['bpmn']:     
        for use_freq_features in [
            True,
        ]:
            task = base_config.task
            time_type = 'relative'
            model = 'ggnn'
            model_config = optimal_hyperparams.OPT_HYPERPARAMS[model]  
            base_config.model = model
            base_config.task = task
            base_config.model_config = model_config
           
            base_config.model_config['add_freq_features'] = use_freq_features
           
            base_config.graph_builder_config['graph_type'] = graph_type
            base_config.run_name = make_experiment_name(model, task, base_config.dataset, None) 
            base_config.group = group_name
            base_config.time_type = time_type   
            base_config.batch_size = 128
            # base_config.model_config['time2vec_k'] = k
            data_module = run_pipeline(base_config, data_module)
            wandb.join()
            wandb.finish()
            gc.collect()


def run_gat_gru(
    base_config,
    group_name, 
):
    for dataset, subset in [
        # ('helpdesk', None),
        ('bpi2017', 'W'),
        ('bpi2017', None),
        ('bpi2012', 'W'),
        ('bpi2012', None),
    ]:
        data_module = None
        for graph_type in ['bpmn']:     
            for use_freq_features in [
                True,
            ]:
                task = base_config.task
                time_type = 'absolute'
                model = 'gat_gru'
                model_config = optimal_hyperparams.OPT_HYPERPARAMS[model]  
                base_config.model = model
                base_config.task = task
                base_config.dataset = dataset
                base_config.process_subset = subset
                base_config.model_config = model_config
            
                base_config.graph_builder_config['graph_type'] = graph_type
                base_config.run_name = make_experiment_name(model, task, base_config.dataset, None) 
                base_config.group = group_name
                base_config.time_type = time_type   
                base_config.batch_size = 128
                # base_config.model_config['time2vec_k'] = k
                data_module = run_pipeline(base_config, data_module)
                wandb.join()
                wandb.finish()
                gc.collect()
    

        
def read_experiment_name():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", default='all')
    parser.add_argument("--dataset", default='helpdesk')
    parser.add_argument("--subset", default=None)
    parser.add_argument("--group_name", default="")
    args = parser.parse_args()
    return args.experiment_name, args.dataset, args.subset, args.group_name

def construct_base_config(dataset_name, subset, experiment_name):
    dataset_to_num_activities = {
        'helpdesk': 16,
        'bpi2017W': 14 + 2,
        'bpi2012': 26 + 2,
        'bpi2012W': 7 + 2,
    }
    base_config = Config.from_config('config.json')
    base_config.dataset = dataset_name
    base_config.process_subset = subset
    base_config.model_config['num_activities'] = dataset_to_num_activities[dataset_name + (subset or "")]
    if experiment_name.startswith('run_g'):
        base_config.preprocessor_slugs.append("StaticGraphBuilder")

    return base_config



def run_experiments(
    experiment_name,
    dataset_name,
    subset,
    group_name,
):
    group_name = 'all' + '_' + group_name
    base_config = construct_base_config(dataset_name, subset, experiment_name)

    if experiment_name == 'all':
        run_sequential_baselines(base_config, group_name)
        base_config.preprocessor_slugs.append("StaticGraphBuilder")
        run_gat_only_with_ablations(base_config, group_name)
        # run_ggnn_only(base_config, group_name)
        # run_gcn(base_config, group_name)
    elif experiment_name == 'run_sequential_baselines':
        run_sequential_baselines(base_config, group_name)
    elif experiment_name == 'run_gcn_only':
        run_gcn_only(base_config, group_name)
    elif experiment_name == 'run_gat_only_with_ablations':
        run_gat_only_with_ablations(base_config, group_name)
    elif experiment_name == 'run_ggnn_only':
        run_ggnn_only(base_config, group_name)
    elif experiment_name == 'run_gat_gru':
        run_gat_gru(base_config, group_name)
    else:
        raise RuntimeWarning(f'Unknown experiment! {experiment_name}')




def main():
    args = read_experiment_name()
    run_experiments(*args)

if __name__ == '__main__':
    main()