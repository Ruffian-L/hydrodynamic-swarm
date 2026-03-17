use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum NodeClass {
    Axiom,
    Observation,
    Hypothesis,
    Directive,
    Anomaly,
}

impl fmt::Display for NodeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeClass::Axiom => write!(f, "[AXIOM]"),
            NodeClass::Observation => write!(f, "[OBSERVATION]"),
            NodeClass::Hypothesis => write!(f, "[HYPOTHESIS]"),
            NodeClass::Directive => write!(f, "[DIRECTIVE]"),
            NodeClass::Anomaly => write!(f, "[ANOMALY]"),
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum RelationalEdge {
    Encapsulates,
    Scaffolds,
    Actuates,
    IsIsomorphicTo,
    Contradicts,
    Catalyzes,
    Synthesizes,
}

impl RelationalEdge {
    pub fn weight(&self) -> i32 {
        match self {
            Self::Encapsulates => 0,
            Self::Scaffolds => 1,
            Self::Actuates => 2,
            Self::IsIsomorphicTo => -5,
            Self::Contradicts => 15,
            Self::Catalyzes => 20,
            Self::Synthesizes => -30,
        }
    }
}

impl fmt::Display for RelationalEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelationalEdge::Encapsulates => write!(f, "[ENCAPSULATES]"),
            RelationalEdge::Scaffolds => write!(f, "[SCAFFOLDS]"),
            RelationalEdge::Actuates => write!(f, "[ACTUATES]"),
            RelationalEdge::IsIsomorphicTo => write!(f, "[IS_ISOMORPHIC_TO]"),
            RelationalEdge::Contradicts => write!(f, "[CONTRADICTS]"),
            RelationalEdge::Catalyzes => write!(f, "[CATALYZES]"),
            RelationalEdge::Synthesizes => write!(f, "[SYNTHESIZES]"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub class: NodeClass,
    pub semantic_hash: String,
    pub embedding: Option<Vec<f32>>,
}

impl Node {
    pub fn new(id: String, class: NodeClass, semantic_hash: String) -> Self {
        Self {
            id,
            class,
            semantic_hash,
            embedding: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub edge: RelationalEdge,
    pub target: String,
    pub justification: Option<String>,
}

impl Edge {
    pub fn new(source: String, edge: RelationalEdge, target: String) -> Self {
        Self {
            source,
            edge,
            target,
            justification: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxTuple {
    pub source: String,
    pub edge: RelationalEdge,
    pub target: String,
}

impl From<Edge> for FluxTuple {
    fn from(edge: Edge) -> Self {
        Self {
            source: edge.source,
            edge: edge.edge,
            target: edge.target,
        }
    }
}
