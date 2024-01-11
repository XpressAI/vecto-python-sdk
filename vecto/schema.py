import sys
from typing import IO, NamedTuple, List, Optional
from datetime import date

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class VectoIngestData(TypedDict):
    '''A named tuple that contains the expected Vecto input format.
    For data, you may use open(path, 'rb') for IMAGE queries or io.StringIO(text) for TEXT queries.
    You may append as many attribute to attributes as needed.'''
    data: IO
    attributes: dict

class VectoEmbeddingData(TypedDict):
    '''A named tuple that contains the expected Vecto embedding format for updating.
    For data, you may use open(path, 'rb') for IMAGE queries or io.StringIO(text) for TEXT queries.
    '''
    id: int
    data: IO

class VectoAttribute(TypedDict):
    '''A named tuple that contains the expected Vecto attribute format for updating.
    You may append as many attribute to attributes as needed'''
    id: int
    attributes: dict

class VectoAnalogyStartEnd(TypedDict):
    '''A named tuple that contains the expected Vecto analogy start-end input format.
    For data, you may use open(path, 'rb') for IMAGE queries or io.StringIO(text) for TEXT queries.'''
    start: IO
    end: IO

class IngestResponse(NamedTuple):
    '''A named tuple that contains a list of ids of ingested data.'''
    ids: List[int]

class LookupResult(NamedTuple):
    '''A named tuple that contains the lookup result content: attributes, id, and similarity.'''
    attributes: object
    id: int
    similarity: float

class VectoModel(NamedTuple):
    '''A named tuple that contains a Vecto model attributes: description, id, modality, name.'''
    description: str
    id: int
    modality: str
    name: str

class VectoVectorSpace(NamedTuple):
    '''A named tuple that contains a Vecto vector space attribute: id, model, name.'''
    id: int
    model: VectoModel
    name: str

class VectoUser(NamedTuple):
    '''A named tuple that contains a Vecto user attribute: id and name.'''
    id: int
    fullName: str

class VectoToken(NamedTuple):
    '''A named tuple that contains Vecto token attributes.'''
    allVectorSpaces: bool
    createdAt: str
    id: int
    name: str
    tokenType: str
    vectorSpaceIds: List[int]

class VectoNewTokenResponse(NamedTuple):
    '''A named tuple that contains a new create a Vecto token response attributes.'''
    accountId: int
    allowsAccessToAllVectorSpaces: bool
    createdAt: str
    id: int
    name: str
    token: str
    tokenType: str
    updatedAt: str
    vectorSpacesIds: List[int]


MODEL_MAP = {
    1: "CLIP",
    2: "SBERT",
    3: "OPENAI"
}


class VectoAnalogy(NamedTuple):
    '''A named tuple that contains a Vecto analogy attributes.'''
    id: int
    name: str
    textAnalogyExampleIds: List
    createdAt: str
    updatedAt: str
    vectorSpaceId: int

class DailyUsageMetric(NamedTuple):
    '''A named tuple that contains daily usage metrics.'''
    date: date
    count: int
    cumulativeCount: int

class UsageMetric(NamedTuple):
    '''A named tuple that contains usage metrics, including an array of daily metrics.'''
    dailyMetrics: List[DailyUsageMetric]

class VectoUsageMetrics(NamedTuple):
    '''A named tuple that contains Vecto usage metrics for lookups and indexing.'''
    lookups: UsageMetric
    indexing: UsageMetric

class MonthlyUsageResponse(NamedTuple):
    '''A named tuple that contains the usage metrics for a specified vector space and month.'''
    year: int
    month: int
    usage: VectoUsageMetrics

class DataEntry(NamedTuple):
    '''A named tuple that represents an individual data entry with an ID and attributes.'''
    id: int
    attributes: dict

class DataPage(NamedTuple):
    '''A named tuple of data entries, contains the total count and a list of 'DataEntry' instances.'''
    count: int
    elements: List[DataEntry]