from __future__ import print_function
from utils.utils import *
from tqdm import tqdm
from model.meta import *
from utils.showbs import *
from collections import OrderedDict
from utils.helpers import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def init_io(root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    io = IOStream('%s/%sway_%sshot_%squery.log' % (root_path,
                                                   args.n_test_way, args.n_test_shot, args.n_test_query))
    return io


def te(args):
    root_path = 'ck_train/%s/%s_%stk_%sbs%sw%ss%sq%s' % (
        args.dataset_train, args.backbone, args.task_num, args.batch_size,args.n_way,args.n_shot, args.n_query, args.exp)
    trainset = get_dataset(args.dataset_train, 'train', args.unsupervised, args, augment=args.augment)
    test_loader,_ = get_test_dataloader(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Meta(args,trainset.__len__())
    path= os.path.join(root_path, "{}.pt".format(args.num_test))

    checkpoint = torch.load(path, map_location='cuda:0')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model

    io.cprint("-----------------------------------------------------------------------------")
    io.cprint('test_num:{:}'.format(args.num_test))
    para_model.eval()
    record = np.zeros(args.n_test_episodes)
    query_labels = torch.arange(args.n_test_way).view(-1, 1).expand(-1, args.n_test_query).reshape(-1).cuda()
    for i, batch in tqdm(enumerate(test_loader, 1), total=len(test_loader), desc='test procedure'):
        if i==1:
            # showsq(args.n_test_way, args.n_test_shot, args.n_test_query, batch[0].detach().clone())
            print(batch[0].size())
        sup_que = batch[0].cuda()
        accs = para_model.finetunning(args, sup_que, query_labels)
        record[i - 1] = accs
        if i % 1000== 0:
            va, vap = compute_confidence_interval(record[:i])
            io.cprint('{} task, {} way {} shot, Test,  acc={:.4f}+{:.4f}'.format(i,args.n_test_way, args.n_test_shot, va,
                                                                                vap))
    va, vap = compute_confidence_interval(record[:i])
    io.cprint('{} task, {} way {} shot, Test,  acc={:.4f}+{:.4f}'.format(i, args.n_test_way, args.n_test_shot, va,
                                                                         vap))
    assert (i == record.shape[0])

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

if __name__ == "__main__":
    args = parse_args()
    args.cuda = torch.cuda.is_available()
    args.num_test="E399"# Best,E399
    args.dataset_train='TieredImageNet' #['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS','CUB''ISIC','EuroSAT','chestx','CropDiseases']
    args.dataset_test='TieredImageNet'
    args.backbone="Res12"# conv4,Res12
    args.task_num=256
    args.batch_size=32 
    args.pmax = 0.6
    args.n_epochs=400
    args.cluster_num =512
    args.exp="_{:}_pmax{:}_cluster_num{:}".format(args.n_epochs,args.pmax,args.cluster_num)
    set_gpu(args.gpu)
    for flag in [1,5]:
        args.n_test_shot = flag
        root_path = 'ck_test/%s_2_%s/%s_%stk_%sbs%sw%ss%sq%s' % (
            args.dataset_train,args.dataset_test, args.backbone, args.task_num, args.batch_size, args.n_way,args.n_shot, args.n_query, args.exp)
        io = init_io(root_path)
        a = te(args)

