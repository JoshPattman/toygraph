package toygraph

type BinaryOp[T, U, V any] interface {
	ComputeForward(aVal *T, bVal *U, resVal *V)
	ComputeGrad(aVal, aGrad *T, bVal, bGrad *U, resVal, resGrad *V)
}

type BinaryNode[T, U, V any] struct {
	viV    Allocator[V]
	op     BinaryOp[T, U, V]
	aEp    Conn[T]
	bEp    Conn[U]
	result V
	grad   V
}

func NewBinaryNode[T, U, V any](g *Graph, viV Allocator[V], op BinaryOp[T, U, V], aEp Conn[T], bEp Conn[U]) *BinaryNode[T, U, V] {
	n := &BinaryNode[T, U, V]{
		viV:    viV,
		op:     op,
		aEp:    aEp,
		bEp:    bEp,
		result: *viV.New(),
		grad:   *viV.New(),
	}
	g.Add(n)
	return n
}

func (n *BinaryNode[T, U, V]) Forward() {
	aRef, _, _ := n.aEp()
	bRef, _, _ := n.bEp()
	n.op.ComputeForward(aRef, bRef, &n.result)
}

func (n *BinaryNode[T, U, V]) Backward() {
	aVal, aGrad, _ := n.aEp()
	bVal, bGrad, _ := n.bEp()
	n.op.ComputeGrad(aVal, aGrad, bVal, bGrad, &n.result, &n.grad)
}

func (n *BinaryNode[T, U, V]) ZeroGrad() {
	n.viV.Zero(&n.grad)
}

func (n *BinaryNode[T, U, V]) C() (val, grad *V, allowed func(*V) error) {
	return &n.result, &n.grad, n.viV.Allowed
}
