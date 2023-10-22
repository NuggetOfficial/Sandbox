# BUILDIN
from dataclasses import dataclass
from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from dateutil import parser as dateparser
import logging

# PACKAGES
from pandas import DataFrame
import json
import requests

SWPC_DATA_REFERER = "https://services.swpc.noaa.gov/json/goes/primary/"
SWPC_SUPPORTED_REQUEST = ['xrays-7-day','xrays-3-day', 'xrays-1-day', 'xrays-6-hour']
SWPC_XRAY_SAT16_HIGHE  = '0.05-0.4nm'
SWPC_XRAY_SAT16_LOWE   = '0.1-0.8nm'

@dataclass
class SWPCDataPoint:
    '''Container for SWPC query entries'''
    time_tag: datetime
    satellite: int
    flux: float
    observed_flux: float
    electron_correction:float
    electron_contaminaton:bool
    energy: str

class SWPCRequest(str):
    def __new__(cls, content: str):
        if content in SWPC_SUPPORTED_REQUEST:
            return super().__new__(cls, content+'.json')
        raise ValueError(f'Query primary/{content} NOT supported')

class SWPCDecoder(json.JSONDecoder):
    '''decode SWPC requests using object hook'''
    def __init__(self, *args, **kwargs) -> None:
        # initialise JSONDecoder
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    # Overwrite object_hook
    def object_hook(self, obj: dict):
        time_tag = obj.get('time_tag')

        if time_tag is not None:
            obj['time_tag'] = dateparser.parse(time_tag)
        
        return SWPCDataPoint(**obj)
    
    @ property
    def dtype(self):
        return SWPCDataPoint

class Requester(ABC):
    @abstractmethod
    def get(self, query: str):...

    @abstractproperty
    def content(self):...

    @abstractproperty 
    def ok(self): ...

class SWPCRequester(Requester):
    '''Allows queries on SWPC content servers'''
    def __init__(self, decoder: json.JSONDecoder=SWPCDecoder) -> None:
        # Initialise from global
        self._REFERER = SWPC_DATA_REFERER

        # Initialise from local
        self.decoder = decoder
        self.dtype   = getattr(self.decoder, 'dtype', dict)

        # Latest response
        self.response = None

    @property
    def content(self):
        # if we have response extract content
        if self.response is None:
            return None
        return self.response.content
    
    @property
    def ok(self):
        # if we have response extract ok
        if self.response is None:
            return None
        return self.response.ok
    
    @property
    def _status_code(self):
        # if we have response extract status_code
        if self.response is None:
            return None
        return self.response.status_code

    def _castDtype(self):
        # Loads bytes with decoder 
        return json.loads(self.content, cls=self.decoder)
    
    def _get(self, query: str):
        # Get at _REFERER
        response = requests.get(self._REFERER + query)
        if response.ok:
            self.response = response
        else:
            logging.warning("REQUEST not ok!")

    def _dataProduct_finalise(self, buffer: list[SWPCDataPoint]) -> tuple[DataFrame]:
        # convert to dataframe
        initialFrame = DataFrame(buffer)

        # groupy by energies and return seperate dataframes
        lowEnergyFrame = initialFrame[initialFrame['energy'] == SWPC_XRAY_SAT16_LOWE]
        highEnergyFrame = initialFrame[initialFrame['energy'] == SWPC_XRAY_SAT16_HIGHE]
     
        # return seperated frames
        return lowEnergyFrame, highEnergyFrame
        
    def get(self, query: SWPCRequest, *, raw: bool=False) -> tuple[DataFrame]:
        # Force dtype
        if isinstance(query, str):
            query = SWPCRequest(query)

        # Get response
        self._get(query=query)

        # if response ok then cast output to dataclass
        if self.ok:
            if raw:
                return self._castDtype()
            else: 
                return self._dataProduct_finalise(self._castDtype())
        
        # otherwise dont load
        return None
    