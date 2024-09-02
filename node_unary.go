package toygraph

type UnaryOp[T, U any] interface {
	ComputeForward(inVal *T, resVal *U)
	ComputeGrad(inVal, inGrad *T, resVal, resGrad *U)
}

type UnaryNode[T, U any] struct {
	alloc  Allocator[U]
	op     UnaryOp[T, U]
	inEp   Conn[T]
	result U
	grad   U
}

func NewUnaryNode[T, U any](g *Graph, alloc Allocator[U], op UnaryOp[T, U], inEp Conn[T]) *UnaryNode[T, U] {
	n := &UnaryNode[T, U]{
		alloc:  alloc,
		op:     op,
		inEp:   inEp,
		result: *alloc.New(),
		grad:   *alloc.New(),
	}
	g.Add(n)
	return n
}

func (n *UnaryNode[T, U]) Forward() {
	valRef, _, _ := n.inEp()
	n.op.ComputeForward(valRef, &n.result)
}

func (n *UnaryNode[T, U]) Backward() {
	inVal, inGrad, _ := n.inEp()
	n.op.ComputeGrad(inVal, inGrad, &n.result, &n.grad)
}

func (n *UnaryNode[T, U]) ZeroGrad() {
	n.alloc.Zero(&n.grad)
}

func (n *UnaryNode[T, U]) C() (val, grad *U, allowed func(*U) error) {
	return &n.result, &n.grad, n.alloc.Allowed
}
