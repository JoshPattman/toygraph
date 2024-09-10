package toygraph

import "slices"

type Node interface {
	Forward()
	Backward()
	ZeroGrad()
}

func Set[T any](conn Conn[T], val T) {
	v, _, allowed := conn()
	if err := allowed(&val); err != nil {
		panic(err)
	}
	*v = val
}

func SetGrad[T any](conn Conn[T], val T) {
	_, g, allowed := conn()
	if err := allowed(&val); err != nil {
		panic(err)
	}
	*g = val
}

func Get[T any](conn Conn[T]) T {
	v, _, _ := conn()
	return *v
}

func GetGrad[T any](conn Conn[T]) T {
	_, g, _ := conn()
	return *g
}

// An endpoint returns a refrence to where the vector is stored
type Conn[T any] func() (value, gradient *T, allowedSet func(*T) error)

type Graph struct {
	nodes []Node
}

func NewGraph() *Graph {
	return &Graph{
		nodes: make([]Node, 0),
	}
}

func (g *Graph) Add(n Node) {
	g.nodes = append(g.nodes, n)
}

func (g *Graph) AllForward() {
	for _, n := range g.nodes {
		n.Forward()
	}
}

func (g *Graph) AllBackward() {
	for _, n := range slices.Backward(g.nodes) {
		n.Backward()
	}
}

func (g *Graph) AllZeroGrads() {
	for _, n := range g.nodes {
		n.ZeroGrad()
	}
}

func (g *Graph) Len() int {
	return len(g.nodes)
}
