import torch
from roost.roost.model import Roost
from roost.roost.data import CompositionData, collate_batch
from roost.core import Normalizer
from torch.utils.data import DataLoader

model_name = 'roost_s-0_t-1'
eval_type="best"
run_id = 9
#device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"
resume = f"../models/{model_name}/{eval_type}-r{run_id}.pth.tar"

checkpoint = torch.load(resume, map_location=device)

model = Roost(**checkpoint["model_params"], device=device,)
# model.to(device)
model.load_state_dict(checkpoint["state_dict"])

batch_size = 128
data_reset = {
            "batch_size": 16 * batch_size,  # faster model inference
            "shuffle": False,  # need fixed data order due to ensembling
        }
workers = 0
data_params = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch,
    }
data_params.update(data_reset)
# def predict(input, model, device):
#     model.to(device)
#     with torch.no_grad():
#         input = input.to(device)
#         out = model(input)
#         _, pre = torch.max(out.data, 1)
#         return pre.item()

#ans = predict(input=,model=model,device=device)

model.eval()

def inference(test_path):
    #test_path = '/home/glard/AML/roost/roost/examples/test.csv'

    fea_path = '/home/glard/AML/roost/roost/data/embeddings/matscholar-embedding.json'
    test_set = CompositionData(data_path=test_path, fea_path=fea_path, task='regression')
    test_set = torch.utils.data.Subset(test_set, range(len(test_set)))
    test_generator = DataLoader(test_set, **data_params)
    with torch.no_grad():
        idx, comp, y_test, output = model.predict(generator=test_generator, )

    normalizer = Normalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])

    pred = normalizer.denorm(output.data)
    y_ensemble = pred.view(-1).numpy()
    return y_ensemble


# out = inference('/home/glard/AML/roost/roost/examples/test.csv')
# print(out)