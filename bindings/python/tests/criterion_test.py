import unittest

import torch
import math 
from gtn.torch import CTCLoss
from torch.autograd import gradcheck

class TestCTCCriterion(unittest.TestCase):
    def test_fwd_trivial(self):
        T = 3
        N = 2
        labels = [[0,0]]
        emissions = torch.FloatTensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).view(1, N, T)
        log_probs = torch.log(emissions)
        fwd = (CTCLoss(log_probs, labels, N-1))
        self.assertAlmostEqual(fwd.item(), 0.0)
    
    def test_fwd(self):
        T = 3
        N = 4
        labels = [[1,2]]
        emissions = torch.FloatTensor([1.0]*T*N).view(1, N, T)
        log_probs = torch.log(emissions)
        fwd = (CTCLoss(log_probs, labels, N-1))
        self.assertAlmostEqual(fwd.item(), -math.log(0.25 * 0.25 * 0.25 * 5))
    
    def test_fwd_bwd(self):
        T = 5
        N = 6
        labels = [[0, 1, 2, 1, 0]]
        emissions = torch.tensor((
            0.633766,  0.221185, 0.0917319, 0.0129757,  0.0142857,  0.0260553,
            0.111121,  0.588392, 0.278779,  0.0055756,  0.00569609, 0.010436,
            0.0357786, 0.633813, 0.321418,  0.00249248, 0.00272882, 0.0037688,
            0.0663296, 0.643849, 0.280111,  0.00283995, 0.0035545,  0.00331533,
            0.458235,  0.396634, 0.123377,  0.00648837, 0.00903441, 0.00623107,
        ), requires_grad=True)
        log_probs = emissions.view(1, N, T).contiguous()
        log_probs = torch.log(log_probs)
        log_probs.retain_grad()
        fwd = (CTCLoss(log_probs, labels, N-1))
        self.assertAlmostEqual(fwd.item(), 3.34211, places=4)
       
        fwd.backward()
        expected_grad = torch.tensor((
            -0.366234, 0.221185,  0.0917319, 0.0129757,  0.0142857,  0.0260553,
            0.111121,  -0.411608, 0.278779,  0.0055756,  0.00569609, 0.010436,
            0.0357786, 0.633813,  -0.678582, 0.00249248, 0.00272882, 0.0037688,
            0.0663296, -0.356151, 0.280111,  0.00283995, 0.0035545,  0.00331533,
            -0.541765, 0.396634,  0.123377,  0.00648837, 0.00903441, 0.00623107,
        )).view(1, N, T)
        self.assertTrue(log_probs.grad.allclose(expected_grad))

    # Jacobian test does not work at fp32 precision
    # def test_jacobian(self):
    #     T = 25
    #     N = 10
    #     B = 5
    #     inputs = torch.randn(B, N, T, dtype=torch.float,requires_grad=True)
    #     m = torch.nn.LogSoftmax(2)
    #     log_probs = m(inputs)
    #     log_probs.retain_grad()
    #     input = (log_probs, [[0,1,2,3] * B], N-1) #,[0],[1,1,1]
    #     test = gradcheck(CTCLoss, input, eps=1e-3, rtol=1e-2, raise_exception=False)
    #     print(test)



if __name__ == '__main__':
    unittest.main()