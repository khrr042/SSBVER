import torch
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from .utils import eval_reid, re_ranking

from tools.rank_cylib.rank_cy import evaluate_cy

def do_eval(args,
            val_loader,
            model,
            logger=None,
            EMA=False):
    
    model.eval()
    features, ids, cams = [], [], []
    num_q = val_loader.dataset.num_q
    with torch.no_grad():
        for in_batch in val_loader:
            images, vids, camids = in_batch
            images = images.cuda()
            feats = model(images)
            if args.test_hflip:
                feats += model(F.hflip(images))
                feats /= 2
            for j in range(feats.shape[0]):
                features.append(feats[j].reshape(feats[j].shape[0]).cpu())
                ids.append(vids[j])
                cams.append(camids[j])

    
    features = torch.stack(features, dim=0)
    features = torch.nn.functional.normalize(features, dim=1, p=2)

    qf = features[:num_q]
    q_vids = np.asarray(ids[:num_q])
    q_cams = np.asarray(cams[:num_q])

    gf = features[num_q:]
    g_vids = np.asarray(ids[num_q:])
    g_cams = np.asarray(cams[num_q:])

    if args.dump_features:
        dump_path = args.feature_dump_path
        if not dump_path:
            dump_path = os.path.join(
                args.output_dir,
                'features_{}_{}.pth'.format(
                    getattr(args, 'test_model', 'model'),
                    args.neck_feat))
        dump_dir = os.path.dirname(dump_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

        torch.save({
            'features': features,
            'qf': qf,
            'gf': gf,
            'q_vids': q_vids,
            'q_cams': q_cams,
            'g_vids': g_vids,
            'g_cams': g_cams,
            'num_q': num_q,
            'model_arc': args.model_arc,
            'test_model': getattr(args, 'test_model', ''),
            'neck_feat': args.neck_feat,
        }, dump_path)
        if logger is not None:
            logger.info('Features are dumped to {}'.format(dump_path))
        else:
            print('Features are dumped to {}'.format(dump_path))
    
    if args.re_rank:
        distmat = re_ranking(qf, 
                            gf, 
                            k1=args.k1, 
                            k2=args.k2, 
                            lambda_value=args.lambda_value)
    else:
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat = torch.addmm(distmat, qf, gf.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()
        distmat = np.power(distmat + 1e-5, 0.5)

    if args.plot_dist:
        mask_id = np.expand_dims(q_vids, 0).T == np.expand_dims(g_vids, 0)
        mask_cam = np.expand_dims(q_cams, 0).T == np.expand_dims(g_cams, 0)
        mask = np.logical_and(mask_id, np.invert(mask_cam))
        if args.dataset == 'VehicleID':
            mask = mask_id
        pos_mean, pos_std = distmat[mask].mean(), distmat[mask].std()
        neg_mean, neg_std = distmat[np.invert(mask_id)].mean(), \
                            distmat[np.invert(mask_id)].std()
        plt.hist(distmat[mask], bins=500, color='r', 
                label='Positive, $\mu$={0:.2f}, $\sigma$={1:.2f}'.format(\
                                        pos_mean, pos_std), density=True)
        plt.hist(distmat[np.invert(mask)], bins=500, color='b', 
                label='Negative, $\mu$={0:.2f}, $\sigma$={1:.2f}'.format(\
                                        neg_mean, neg_std), density=True)
        plt.legend()
        plt.grid(True)
        plt.xlabel('$L_2$ Feature Distance')
        plt.ylabel('Density')
        plt.savefig('./results/{}_{}_SSBVER.jpg'.format(args.dataset, 
                                args.model_arc), dpi=600, bbox_inches='tight')


    if logger is not None:
        logger.info('computing metrics ...')
        if args.cython_eval:
            cmc, mAP = evaluate_cy(distmat, q_vids, g_vids, q_cams, g_cams, 50)
        else:
            cmc, mAP = eval_reid(distmat, q_vids, g_vids, q_cams, g_cams)
        if EMA:
            logger.info('Teacher Validation Results:')
        else:
            logger.info('Student Validation Results:')
        logger.info("mAP: {:.2%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    else:
        print('computing metrics ...')
        if args.cython_eval:
            cmc, mAP = evaluate_cy(distmat, q_vids, g_vids, q_cams, g_cams, 50)
        else:
            cmc, mAP = eval_reid(distmat, q_vids, g_vids, q_cams, g_cams)
        print('Validation Results:')
        print("mAP: {:.2%}".format(mAP))
        for r in [1, 5, 10]:
            print("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    
    return mAP, cmc[0], cmc[4] 
                
