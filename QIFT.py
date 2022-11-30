import numpy as np
import opt_einsum as oe
import time

np.set_printoptions(linewidth=500)

def Phase(theta):
    return np.array([[1, 0], [0, np.exp(1j*theta)]]).astype(np.complex128)

def exp_theta(theta):
    return np.array([[1, 1], [1, np.exp(1j*theta)]]).astype(np.complex128)

def order3_Phase(theta):
    return oe.contract("ab,bc->abc", np.eye(2), exp_theta(theta)).reshape(2,2,2,1)

def order4_Phase(theta):
    return oe.contract("ab,cd,bc->abcd", np.eye(2), np.eye(2), exp_theta(theta))

def copy_H():
    H = np.array([[1,1],[1,-1]]).astype(np.complex128) / np.sqrt(2)
    copy_tensor = np.zeros((2,2,2)).astype(np.complex128)
    copy_tensor[0,0,0] = copy_tensor[1,1,1] = 1.0
    return oe.contract("aBc,bB->abc", copy_tensor, H).reshape(2,2,1,2)

def prepare_MPO(n):
    if n == 1:
        MPO = [np.array([[1,1],[1,-1]]).astype(np.complex128).reshape(2,2,1,1) / np.sqrt(2)]
        return MPO
    MPO = [copy_H()]
    for i in range(1, n-1):
        MPO.append(order4_Phase(np.pi/(2**i)))
    MPO.append(order3_Phase(np.pi/(2**(n-1))))
    return MPO

def contract_MPO(MPO1, MPO2, chi):
    newMPO = []
    cdim, gdim = MPO1[-1].shape[2], MPO2[-1].shape[2]
    bottom = oe.contract("abcd,eagh->gcebdh", MPO1[-1], MPO2[-1]).reshape(-1, 4)
    U, s, Vh = np.linalg.svd(bottom, full_matrices=False)
    if len(s) < chi:
        newMPO.append(Vh.reshape(-1,2,2,1).transpose(1,2,0,3))
        tmp = np.dot(U, np.diag(s)).reshape(gdim, cdim, -1)
    else:
        newMPO.append(Vh[:chi].reshape(-1,2,2,1).transpose(1,2,0,3))
        tmp = np.dot(U[:,:chi], np.diag(s[:chi])).reshape(gdim, cdim, -1)

    for i in range(1, len(MPO2)):
        dimi = tmp.shape[2]
        cdim, gdim = MPO1[-1-i].shape[2], MPO2[-1-i].shape[2]
        tmp = oe.contract("abcd,eagh,hdi->gcebi", MPO1[-1-i], MPO2[-1-i], tmp).reshape(-1, 4*dimi)
        U, s, Vh = np.linalg.svd(tmp, full_matrices=False)
        if len(s) < chi:
            newMPO.append(Vh.reshape(-1,2,2,dimi).transpose(1,2,0,3))
            tmp = np.dot(U, np.diag(s)).reshape(gdim, cdim, -1)
        else:
            newMPO.append(Vh[:chi].reshape(-1,2,2,dimi).transpose(1,2,0,3))
            tmp = np.dot(U[:,:chi], np.diag(s[:chi])).reshape(gdim, cdim, -1)
    
    dimd = newMPO[-1].shape[3]
    newMPO[-1] = oe.contract("abcd,efc->abefd", newMPO[-1], tmp).reshape(2,2,-1,dimd)
    
    # move apex to bottom
    for i in range(len(MPO2)-1):
        dimc, dimd = newMPO[-1-i].shape[2], newMPO[-1-i].shape[3]
        tmp = oe.contract("abcd,efdg->abcefg", newMPO[-1-i], newMPO[-2-i]).reshape(4*dimc,-1)
        U, s, Vh = np.linalg.svd(tmp, full_matrices=False)
        newMPO[-1-i] = U[:,:dimd].reshape(2,2,dimc,dimd)
        newMPO[-2-i] = np.dot(np.diag(s[:dimd]), Vh[:dimd]).reshape(dimd,2,2,-1).transpose(1,2,0,3)

    return MPO1[:len(MPO1) - len(MPO2)] + newMPO[::-1]

def convert_QFT_to_MPO(n, chi):
    MPO1 = prepare_MPO(n)
    for i in range(1, n):
        MPO2 = prepare_MPO(n-i)
        MPO1 = contract_MPO(MPO1, MPO2, chi)
    return MPO1

def qift(MPS, QFT_MPO):
    """Quantum-inspired Fourier Transformation

    Args:
        MPS (List[np.array]) : the input MPS. Each tensor has a shape (2, *, *)
        QFT_MPO(List[np.array]) : the MPO that represents Q_n in QFT

    Return:
        MPS (List[np.array]) : the result MPS.
    """
    n = len(MPS)
    if MPS[0].shape[1] != 1:
        raise ValueError("the left dimension of MPS must be 1.")
    if MPS[-1].shape[2] !=1:
        raise ValueError("the right dimension of MPS must be 1.")
    for i in range(n-1):
        if MPS[i].shape[2] != MPS[i+1].shape[1]:
            raise ValueError(f"the bond dimension between {i} and {i+1} differs.")

    res = []
    for i in range(n):
        left_dim = QFT_MPO[i].shape[2] * MPS[i].shape[1]
        res.append(oe.contract("abcd,bef->acedf", QFT_MPO[i], MPS[i]).reshape(2, left_dim, -1))
    return res

def initialize_input_MPS(n, chi, seed):
    np.random.seed(seed)
    if n == 1:
        return [np.random.randn(2,1,1)]
    MPS = [np.random.randn(2,1,chi)]
    for i in range(1, n-1):
        MPS.append(np.random.randn(2,chi,chi))
    MPS.append(np.random.randn(2,chi,1))
    x = convert_MPS_to_state(MPS)
    MPS[0] /= np.linalg.norm(x)
    return MPS

def convert_MPS_to_state(MPS):
    new_MPS = []
    while len(new_MPS) != 1:
        new_MPS = []
        for i in range(0, len(MPS)-1, 2):
            dimb, dime = MPS[i].shape[1], MPS[i+1].shape[2]
            new_MPS.append(oe.contract("abc,dce->adbe", MPS[i], MPS[i+1]).reshape(-1,dimb,dime))
        if len(MPS) % 2 == 1:
            new_MPS.append(MPS[-1])
        MPS = new_MPS
    res = new_MPS[0].flatten()
    return res
        
def correct_qubit_reversal(x):
    transpose_list = []
    n = int(np.log2(len(x)))
    for i in range(2**n):
        transpose_list.append(int(f"{i:0{n}b}"[::-1], 2))
    return x[transpose_list]

if __name__ == "__main__":
    for n in range(10, 25, 2):
        print(f"num of qubits: {n}")
        mps_x = initialize_input_MPS(n, 2, n)

        # fft in numpy
        start = time.time()
        x = convert_MPS_to_state(mps_x)
        y = np.fft.ifft(x, norm="ortho")
        time_fft = time.time() - start

        # prepare mpo that represents Q_n in QFT
        qft_mpo = convert_QFT_to_MPO(n, 8)

        # qift
        start = time.time()
        mps = qift(mps_x, qft_mpo)
        yqift = convert_MPS_to_state(mps)
        end = time.time()
        time_qift = end - start 

        # verify that the two results agree.
        yqift = correct_qubit_reversal(yqift)
        assert np.allclose(y, yqift)

        print(f"elapsed time for fft: {time_fft} qift: {time_qift}")