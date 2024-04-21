import torch
import utility
from decimal import Decimal
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            sr = self.model(lr[0])
            sr2lr = []  # 由生成的sr通过降采样转换回的低分辨率
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])  # 从后向前访问
                sr2lr.append(sr2lr_i)

            # compute primary loss
            loss_primary = self.loss(sr[-1], hr)
            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

            # compute dual loss
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])

            # compute total loss
            loss = loss_primary + self.opt.dual_weight * loss_dual
            # 如果当前批次的损失值小于一个阈值继续进行反向传播，否则跳过训练批次
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()                
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]  # 获取日志中最后元素的最后一个值
        self.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        # self.ckp.add_log(torch.zeros(1, 1))
        self.ckp.add_log(torch.zeros(1, 2))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                eval_ssim = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)  # tqdm函数创建了一个带有进度条的迭代器，进度条的宽度为80个字符
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]  # 文件名
                    no_eval = (hr.nelement() == 1)  # 如果hr的元素为1时无法进行评估，不为1时说明为有标签数据，可以进行评估
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)  # 此时返回的是一个包含单个元素的元组

                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]  # 判断sr是否为列表，如果是则把sr的最后一个元素赋给sr变量
                    sr = utility.quantize(sr, self.opt.rgb_range)  # 对sr进行量化

                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,  # s是尺度
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_ssim += utility.calc_ssim(
                            sr, hr, s, self.opt.rgb_range,  # s是尺度
                            benchmark=self.loader_test.dataset.benchmark
                        )
                    # print(self.opt.save_results)
                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)

                # self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                self.ckp.log[-1] = torch.tensor([eval_psnr / len(self.loader_test), eval_ssim / len(self.loader_test)])


                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f}, SSIM: {:.4f} (Best PSNR: {:.2f} @epoch {}, SSIM: {:.4f})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1][0],
                        self.ckp.log[-1][1],
                        best[0][0],
                        best[1][0] + 1,
                        best[0][1]
                    )
                )

                # self.ckp.write_log(
                #     '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                #         self.opt.data_test, s,
                #         self.ckp.log[-1, si],
                #         best[0][si],
                #         best[1][si] + 1
                #     )
                # )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]], 

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
