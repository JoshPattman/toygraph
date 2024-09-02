package toygraph

import "math"

type Num interface {
	float64 | float32 | int
}

// *************************************************** Allocators ***************************************************
type numAllocator[T Num] struct{}

func (v *numAllocator[T]) New() *T {
	return new(T)
}
func (v *numAllocator[T]) Zero(t *T) {
	*t = 0
}

func (v *numAllocator[T]) Allowed(*T) error {
	return nil
}

// *************************************************** Inputs ***************************************************
func NumValue[T Num](g *Graph) *ValueNode[T] {
	return NewValueNode(g, &numAllocator[T]{})
}

// *************************************************** Unary Ops ***************************************************
type numExpOp[T Num] struct{}

func NumExp[T Num](g *Graph, inEp Conn[T]) *UnaryNode[T, T] {
	return NewUnaryNode(g, &numAllocator[T]{}, &numExpOp[T]{}, inEp)
}

func (o *numExpOp[T]) ComputeForward(inVal *T, resVal *T) {
	*resVal = T(math.Exp(float64(*inVal)))
}

func (o *numExpOp[T]) ComputeGrad(inVal, inGrad *T, resVal, resGrad *T) {
	*inGrad += T(math.Exp(float64(*inVal))) * *resGrad
}

type numSinOp[T Num] struct{}

func NumSin[T Num](g *Graph, inEp Conn[T]) *UnaryNode[T, T] {
	return NewUnaryNode(g, &numAllocator[T]{}, &numSinOp[T]{}, inEp)
}

func (o *numSinOp[T]) ComputeForward(inVal *T, resVal *T) {
	*resVal = T(math.Sin(float64(*inVal)))
}

func (o *numSinOp[T]) ComputeGrad(inVal, inGrad *T, resVal, resGrad *T) {
	*inGrad += T(math.Cos(float64(*inVal))) * *resGrad
}

type numCosOp[T Num] struct{}

func NumCos[T Num](g *Graph, inEp Conn[T]) *UnaryNode[T, T] {
	return NewUnaryNode(g, &numAllocator[T]{}, &numCosOp[T]{}, inEp)
}

func (o *numCosOp[T]) ComputeForward(inVal *T, resVal *T) {
	*resVal = T(math.Cos(float64(*inVal)))
}

func (o *numCosOp[T]) ComputeGrad(inVal, inGrad *T, resVal, resGrad *T) {
	*inGrad += T(-math.Sin(float64(*inVal))) * *resGrad
}

// *************************************************** Binary Ops ***************************************************

type numAddOp[T Num] struct{}

func NumAdd[T Num](g *Graph, inEpA Conn[T], inEpB Conn[T]) *BinaryNode[T, T, T] {
	return NewBinaryNode(g, &numAllocator[T]{}, &numAddOp[T]{}, inEpA, inEpB)
}

func (o *numAddOp[T]) ComputeForward(aVal, bVal, resVal *T) {
	*resVal = *aVal + *bVal
}

func (o *numAddOp[T]) ComputeGrad(aVal, aGrad, bVal, bGrad, resVal, resGrad *T) {
	*aGrad += *resGrad
	*bGrad += *resGrad
}

type numMulOp[T Num] struct{}

func NumMul[T Num](g *Graph, inEpA Conn[T], inEpB Conn[T]) *BinaryNode[T, T, T] {
	return NewBinaryNode(g, &numAllocator[T]{}, &numMulOp[T]{}, inEpA, inEpB)
}

func (o *numMulOp[T]) ComputeForward(aVal, bVal, resVal *T) {
	*resVal = *aVal * *bVal
}

func (o *numMulOp[T]) ComputeGrad(aVal, aGrad, bVal, bGrad, resVal, resGrad *T) {
	*aGrad += *resGrad * *bVal
	*bGrad += *resGrad * *aVal
}
