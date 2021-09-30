考虑带权的information networks.

* Frist-order Proximity

pairwise proximity

$w_{uv}$ indicates the frist-order proximity between $u$ and $v$. no edges, frist-order proximity is 0

* Second-order Proximity

let $p_u=(w_{u,1},...,w_{u,|V|})$ denote the frist-order proximity of $u$ with all the other vertices.

second-order proximity between $u$ and $v$ is determined by similarity between $p_u$ and $p_v$。

Several requirements:

* Preserve both the first-order proximity and the second-order proximity between the vertices.
* Scale for very large networks.
* Deal with arbitrary types of edges: directed, undirected and /or weighted.



