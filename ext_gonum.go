package toygraph

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func matLike(conn Conn[*mat.Dense]) *mat.Dense {
	r, c := Get(conn).Dims()
	return mat.NewDense(r, c, nil)
}

func vecLike(conn Conn[*mat.VecDense]) *mat.VecDense {
	l := Get(conn).Len()
	return mat.NewVecDense(l, nil)
}

// *************************************************** Allocators ***************************************************

var _ Allocator[*mat.VecDense] = &vecAllocator{}

type vecAllocator struct {
	Size int
}

// NewVecAllocatorLike creates a new [Allocator] which actos on vectors of the same length as v
func NewVecAllocatorLike(v *mat.VecDense) *vecAllocator {
	return &vecAllocator{
		Size: v.Len(),
	}
}

func NewVecAllocator(n int) *vecAllocator {
	return &vecAllocator{
		Size: n,
	}
}

func (v *vecAllocator) New() **mat.VecDense {
	vec := mat.NewVecDense(v.Size, nil)
	return &vec
}
func (v *vecAllocator) Zero(t **mat.VecDense) {
	(*t).Zero()
}

func (v *vecAllocator) Allowed(m **mat.VecDense) error {
	if v.Size != (*m).Len() {
		return fmt.Errorf("cannot set vector, expected length of %d but was given %d", v.Size, (*m).Len())
	}
	return nil
}

var _ Allocator[*mat.Dense] = &matAllocator{}

type matAllocator struct {
	Rows, Cols int
}

// NewMatAllocatorLike creates a new [Allocator] which actos on matricies of the same shape as m
func NewMatAllocatorLike(m *mat.Dense) *matAllocator {
	r, c := m.Dims()
	return &matAllocator{
		Rows: r,
		Cols: c,
	}
}

func (a *matAllocator) New() **mat.Dense {
	m := mat.NewDense(a.Rows, a.Cols, nil)
	return &m
}
func (a *matAllocator) Zero(t **mat.Dense) {
	(*t).Zero()
}

func (v *matAllocator) Allowed(m **mat.Dense) error {
	r, c := (*m).Dims()
	if !(r == v.Rows && c == v.Cols) {
		return fmt.Errorf("cannot set matrix, expected dims of %d,%d but was given %d,%d", v.Rows, v.Cols, r, c)
	}
	return nil
}

// *************************************************** Inputs ***************************************************

// VecValue creates a new [ValueNode] that holds a vector of length size
func VecValue(g *Graph, size int) *ValueNode[*mat.VecDense] {
	return NewValueNode(g, &vecAllocator{Size: size})
}

// MatValue creates a new [ValueNode] that holds a matrix of dims rows, cols
func MatValue(g *Graph, rows, cols int) *ValueNode[*mat.Dense] {
	return NewValueNode(g, &matAllocator{Rows: rows, Cols: cols})
}

// *************************************************** Unary Ops ***************************************************
type matElemOp struct {
	op UnaryOp[float64, float64]
}

func matElemUnary(g *Graph, op UnaryOp[float64, float64], inEp Conn[*mat.Dense]) *UnaryNode[*mat.Dense, *mat.Dense] {
	return NewUnaryNode(
		g,
		NewMatAllocatorLike(Get(inEp)),
		&matElemOp{op: op},
		inEp,
	)
}

// MatExpElem creates a node that performs an element-wise e^x operation
func MatExpElem(g *Graph, inEp Conn[*mat.Dense]) *UnaryNode[*mat.Dense, *mat.Dense] {
	return matElemUnary(g, &numExpOp[float64]{}, inEp)
}

// MatSinElem creates a node that performs an element-wise sin operation
func MatSinElem(g *Graph, inEp Conn[*mat.Dense]) *UnaryNode[*mat.Dense, *mat.Dense] {
	return matElemUnary(g, &numSinOp[float64]{}, inEp)
}

// MatCosElem creates a node that performs an element-wise cos operation
func MatCosElem(g *Graph, inEp Conn[*mat.Dense]) *UnaryNode[*mat.Dense, *mat.Dense] {
	return matElemUnary(g, &numCosOp[float64]{}, inEp)
}

func (o *matElemOp) ComputeForward(inVal, resVal **mat.Dense) {
	rdIn := (*inVal).RawMatrix().Data
	rdRes := (*resVal).RawMatrix().Data
	for i := range rdIn {
		o.op.ComputeForward(&rdIn[i], &rdRes[i])
	}
}
func (o *matElemOp) ComputeGrad(inVal, inGrad, resVal, resGrad **mat.Dense) {
	rdIn := (*inVal).RawMatrix().Data
	rdRes := (*resVal).RawMatrix().Data
	rdGradIn := (*inGrad).RawMatrix().Data
	rdGradRes := (*resGrad).RawMatrix().Data
	for i := range rdIn {
		o.op.ComputeGrad(&rdIn[i], &rdGradIn[i], &rdRes[i], &rdGradRes[i])
	}
}

type vecElemOp struct {
	op UnaryOp[float64, float64]
}

func vecElemUnary(g *Graph, op UnaryOp[float64, float64], inEp Conn[*mat.VecDense]) *UnaryNode[*mat.VecDense, *mat.VecDense] {
	return NewUnaryNode(
		g,
		NewVecAllocatorLike(Get(inEp)),
		&vecElemOp{op: op},
		inEp,
	)
}

// VecExpElem creates a node that performs an element-wise e^x operation
func VecExpElem(g *Graph, inEp Conn[*mat.VecDense]) *UnaryNode[*mat.VecDense, *mat.VecDense] {
	return vecElemUnary(g, &numExpOp[float64]{}, inEp)
}

// VecSinElem creates a node that performs an element-wise sin operation
func VecSinElem(g *Graph, inEp Conn[*mat.VecDense]) *UnaryNode[*mat.VecDense, *mat.VecDense] {
	return vecElemUnary(g, &numSinOp[float64]{}, inEp)
}

// VecCosElem creates a node that performs an element-wise cos operation
func VecCosElem(g *Graph, inEp Conn[*mat.VecDense]) *UnaryNode[*mat.VecDense, *mat.VecDense] {
	return vecElemUnary(g, &numCosOp[float64]{}, inEp)
}

func (o *vecElemOp) ComputeForward(inVal, resVal **mat.VecDense) {
	rdIn := (*inVal).RawVector().Data
	rdRes := (*resVal).RawVector().Data
	for i := range rdIn {
		o.op.ComputeForward(&rdIn[i], &rdRes[i])
	}
}
func (o *vecElemOp) ComputeGrad(inVal, inGrad, resVal, resGrad **mat.VecDense) {
	rdIn := (*inVal).RawVector().Data
	rdRes := (*resVal).RawVector().Data
	rdGradIn := (*inGrad).RawVector().Data
	rdGradRes := (*resGrad).RawVector().Data
	for i := range rdIn {
		o.op.ComputeGrad(&rdIn[i], &rdGradIn[i], &rdRes[i], &rdGradRes[i])
	}
}

// *************************************************** Binary Ops ***************************************************
type matAddOp struct{}

// MatAddElem creates a node that performs an element-wise add operation
func MatAddElem(g *Graph, inEpA Conn[*mat.Dense], inEpB Conn[*mat.Dense]) *BinaryNode[*mat.Dense, *mat.Dense, *mat.Dense] {
	return NewBinaryNode(g, NewMatAllocatorLike(Get(inEpA)), &matAddOp{}, inEpA, inEpB)
}

func (o *matAddOp) ComputeForward(aVal, bVal, resVal **mat.Dense) {
	(*resVal).Add(*aVal, *bVal)
}

func (o *matAddOp) ComputeGrad(aVal, aGrad, bVal, bGrad, resVal, resGrad **mat.Dense) {
	(*aGrad).Add(*aGrad, *resGrad)
	(*bGrad).Add(*bGrad, *resGrad)
}

type vecAddOp struct{}

func VecAddElem(g *Graph, inEpA Conn[*mat.VecDense], inEpB Conn[*mat.VecDense]) *BinaryNode[*mat.VecDense, *mat.VecDense, *mat.VecDense] {
	return NewBinaryNode(g, NewVecAllocatorLike(Get(inEpA)), &vecAddOp{}, inEpA, inEpB)
}

func (o *vecAddOp) ComputeForward(aVal, bVal, resVal **mat.VecDense) {
	(*resVal).AddVec(*aVal, *bVal)
}

func (o *vecAddOp) ComputeGrad(aVal, aGrad, bVal, bGrad, resVal, resGrad **mat.VecDense) {
	(*aGrad).AddVec(*aGrad, *resGrad)
	(*bGrad).AddVec(*bGrad, *resGrad)
}

type matMulElemOp struct {
	tempBuf *mat.Dense
}

func MatMulElem(g *Graph, inEpA Conn[*mat.Dense], inEpB Conn[*mat.Dense]) *BinaryNode[*mat.Dense, *mat.Dense, *mat.Dense] {
	return NewBinaryNode(
		g,
		NewMatAllocatorLike(Get(inEpA)),
		&matMulElemOp{tempBuf: matLike(inEpA)},
		inEpA,
		inEpB,
	)
}

func (o *matMulElemOp) ComputeForward(aVal, bVal, resVal **mat.Dense) {
	(*resVal).MulElem(*aVal, *bVal)
}

func (o *matMulElemOp) ComputeGrad(aVal, aGrad, bVal, bGrad, resVal, resGrad **mat.Dense) {
	// Grad for a
	o.tempBuf.MulElem(*resGrad, *bVal)
	(*aGrad).Add(*aGrad, o.tempBuf)
	// Grad for b
	o.tempBuf.MulElem(*resGrad, *aVal)
	(*bGrad).Add(*bGrad, o.tempBuf)
}

type veMulElemOp struct {
	tempBuf *mat.VecDense
}

func VecMulElem(g *Graph, inEpA Conn[*mat.VecDense], inEpB Conn[*mat.VecDense]) *BinaryNode[*mat.VecDense, *mat.VecDense, *mat.VecDense] {
	return NewBinaryNode(
		g,
		NewVecAllocatorLike(Get(inEpA)),
		&veMulElemOp{tempBuf: vecLike(inEpA)},
		inEpA,
		inEpB,
	)
}

func (o *veMulElemOp) ComputeForward(aVal, bVal, resVal **mat.VecDense) {
	(*resVal).MulElemVec(*aVal, *bVal)
}

func (o *veMulElemOp) ComputeGrad(aVal, aGrad, bVal, bGrad, resVal, resGrad **mat.VecDense) {
	// Grad for a
	o.tempBuf.MulElemVec(*resGrad, *bVal)
	(*aGrad).AddVec(*aGrad, o.tempBuf)
	// Grad for b
	o.tempBuf.MulElemVec(*resGrad, *aVal)
	(*bGrad).AddVec(*bGrad, o.tempBuf)
}

type matMulVecOp struct {
	aGradBuf *mat.Dense
	bGradBuf *mat.VecDense
}

func MatMulVec(g *Graph, inEpA Conn[*mat.Dense], inEpB Conn[*mat.VecDense]) *BinaryNode[*mat.Dense, *mat.VecDense, *mat.VecDense] {
	r, _ := Get(inEpA).Dims()
	return NewBinaryNode(
		g,
		NewVecAllocator(r),
		&matMulVecOp{
			aGradBuf: matLike(inEpA),
			bGradBuf: vecLike(inEpB),
		},
		inEpA,
		inEpB,
	)
}

func (o *matMulVecOp) ComputeForward(aVal **mat.Dense, bVal, resVal **mat.VecDense) {
	// Multiply matrix aVal by vector bVal and put result in resVal
	(*resVal).MulVec(*aVal, *bVal)
}

func (o *matMulVecOp) ComputeGrad(aVal, aGrad **mat.Dense, bVal, bGrad, resVal, resGrad **mat.VecDense) {
	// grad for vector
	o.bGradBuf.MulVec((*aVal).T(), *resGrad)
	(*bGrad).AddVec(*bGrad, o.bGradBuf)
	// grad for matrix
	o.aGradBuf.Outer(1, *resGrad, *bVal)
	(*aGrad).Add(*aGrad, o.aGradBuf)

}
