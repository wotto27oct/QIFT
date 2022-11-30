from QIFT import *

def convert_MPO_to_op(MPO):
    left_tensor = MPO[0]
    dimc = left_tensor.shape[2]
    for i, tensor in enumerate(MPO[1:], 2):
        left_tensor = oe.contract("abcd,efdg->aebfcg", left_tensor, tensor).reshape(2**i,2**i,dimc,-1)
    return left_tensor.reshape(2**len(MPO), 2**len(MPO), dimc, -1)

def reversal_op(n):
    mat = np.zeros((2**n,2**n))
    for i in range(2**n):
        idx = int(f"{i:0{n}b}"[::-1], 2)
        mat[i][idx] = 1
    return mat

def true_Fn(n):
    mat = np.zeros((2**n, 2**n)).astype(np.complex128)
    omega = np.exp(2*np.pi*1j/(2**n))
    for i in range(2**n):
        for j in range(2**n):
            mat[i,j] = omega**(i*j)
    return mat / np.sqrt(2**n)

def test_contract_MPO(n):
    assert n <= 10
    MPO1 = prepare_MPO(n)
    MPO2 = prepare_MPO(n-1)
    op1 = convert_MPO_to_op(MPO1).reshape(2**n, -1)
    op2 = np.kron(np.eye(2), convert_MPO_to_op(MPO2).reshape(2**(n-1),-1))
    op3 = np.dot(op2, op1)
    MPO1 = contract_MPO(MPO1, MPO2, 8)
    op4 = convert_MPO_to_op(MPO1).reshape(2**n,2**n)
    assert np.allclose(op3, op4)

def test_QFT_matrix(n):
    assert n <= 10
    QFT_MPO = convert_QFT_to_MPO(n, 8)
    Qn = convert_MPO_to_op(QFT_MPO).reshape(2**n, -1)
    Rn = reversal_op(n)
    QFT = np.dot(Rn, Qn)

    true_QFT = true_Fn(n)

    assert np.allclose(QFT, true_QFT)

if __name__ == "__main__":
    for i in range(2, 11):
        print(i)
        test_contract_MPO(i)
        test_QFT_matrix(i)