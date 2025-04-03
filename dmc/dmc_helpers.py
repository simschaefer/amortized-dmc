import pandas as pd
import numpy as np
import time

def format_empirical_data(data):
    
    # extract relveant variables
    data_np = data[['rt', 'accuracy', "congruency_num"]].values

    # convert to dictionary
    inference_data = dict(rt=data_np[:,0],
                          accuracy=data_np[:,1],
                          conditions=data_np[:,2])

    # add dimensions so it fits training data
    inference_data = {k: v[np.newaxis,..., np.newaxis] for k, v in inference_data.items()}

    # adjust dimensions of num_obs
    inference_data["num_obs"] = np.array([data_np.shape[0]])[:,np.newaxis]
    
    return inference_data


def fit_empirical_data(data, approximator):

    participants=data["participant"].unique()

    list_data_samples=[]

    for i, part in enumerate(participants):
        
        part_data = data[data["participant"]==part]
        
        part_data = format_empirical_data(part_data)
        
        start_time=time.time()
        samples = approximator.sample(conditions=part_data, num_samples=1000)
        end_time=time.time()
        
        sampling_time=end_time-start_time

        samples_2d={k: v.flatten() for k, v in samples.items()}
        
        data_samples=pd.DataFrame(samples_2d)
        
        data_samples["participant"]=part
        data_samples["sampling_time"]=sampling_time
        
        list_data_samples.append(data_samples)

    data_samples_complete=pd.concat(list_data_samples)

    return data_samples_complete
    