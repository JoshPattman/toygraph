package toygraph

type ValueNode[T any] struct {
	vi   Allocator[T]
	val  T
	grad T
}

func NewValueNode[T any](g *Graph, vi Allocator[T]) *ValueNode[T] {
	n := &ValueNode[T]{
		vi:   vi,
		val:  *vi.New(),
		grad: *vi.New(),
	}
	g.Add(n)
	return n
}

func (n *ValueNode[T]) C() (value, gradient *T, allowed func(*T) error) {
	return &n.val, &n.grad, n.vi.Allowed
}

func (n *ValueNode[T]) Forward() {}

func (n *ValueNode[T]) Backward() {}

func (n *ValueNode[T]) ZeroGrad() {
	n.vi.Zero(&n.grad)
}
