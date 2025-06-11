import enum
from typing import Optional, Union, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

# Classes with Enums for constrained values

class UnstructedField(str, enum.Enum):
    INSTRUCTION = "instruction"
    RESPONSE = "response"

class StructedField(str, enum.Enum):
    CATEGORY = "category"
    INTENT = "intent"

class CategoryClass(str, enum.Enum):
     ORDER = 'ORDER'
     SHIPING = 'SHIPPING'
     CANCEL = 'CANCEL'
     INVOICE = 'INVOICE'
     PAYMENT = 'PAYMENT'
     REFUND =  'REFUND'
     FEEDBACK = 'FEEDBACK'
     CONTACT = 'CONTACT'
     ACCOUNT = 'ACCOUNT'
     DELIVERY = 'DELIVERY'
     SUBSCRIPTION = 'SUBSCRIPTION'

class IntentClass(str, enum.Enum):
    CANCEL_ORDER = 'cancel_order'
    CHANGE_ORDER = 'change_order'
    CHANGE_SHIPPING_ADDRESS = 'change_shipping_address'
    CHECK_CANCELLATION_FEE = 'check_cancellation_fee'
    CHECK_INVOICE = 'check_invoice'
    CHECK_PAYMENT_METHODS = 'check_payment_methods'
    CHECK_REFUND_POLICY = 'check_refund_policy'
    COMPLAINT = 'complaint'
    CONTACT_CUSTOMER_SERVICE = 'contact_customer_service'
    CONTACT_HUMAN_AGENT = 'contact_human_agent'
    CREATE_ACCOUNT = 'create_account'
    DELETE_ACCOUNT = 'delete_account'
    DELIVERY_OPTIONS = 'delivery_options'
    DELIVERY_PERIOD = 'delivery_period'
    EDIT_ACCOUNT = 'edit_account'
    GET_INVOICE = 'get_invoice'
    GET_REFUND = 'get_refund'
    NEWSLETTER_SUBSCRIPTION = 'newsletter_subscription'
    PAYMENT_ISSUE = 'payment_issue'
    PLACE_ORDER = 'place_order'
    RECOVER_PASSWORD = 'recover_password'
    REGISTRATION_PROBLEMS = 'registration_problems'
    REVIEW = 'review'
    SET_UP_SHIPPING_ADDRESS = 'set_up_shipping_address'
    SWITCH_ACCOUNT = 'switch_account'
    TRACK_ORDER = 'track_order'
    TRACK_REFUND = 'track_refund'

# Pydantic models for function inputs

class DatasetOverview(BaseModel):
    function_type: Literal["get_dataset_overview"]

class SelectSemanticIntent(BaseModel):
    function_type: Literal["select_semantic_intent"]
    query: str

class SelectSemanticCategory(BaseModel):
    function_type: Literal["select_semantic_category"]
    query: str

class CountIntent(BaseModel):
    function_type: Literal["count_intent"]
    intent_class: IntentClass

class CountCategory(BaseModel):
    function_type: Literal["count_category"]
    category_class: CategoryClass

class SumValues(BaseModel):
    function_type: Literal["sum_values"]
    values: List[float]

class MultiplicationFloat(BaseModel):
    function_type: Literal["multiplication_float"]
    a: float
    b: float

class DivisionFloat(BaseModel):
    function_type: Literal["division_float"]
    numerator: float
    denominator: float

    @model_validator(mode="after")
    def check_denominator_not_zero(self) -> "DivisionFloat":
        if self.denominator == 0:
            raise ValueError("Denominator cannot be zero")
        return self

class ShowExamples(BaseModel):
    function_type: Literal["show_examples"]
    target_field: Optional[UnstructedField] = None
    condition_field: Optional[StructedField] = None
    condition_field_value: Optional[Union[CategoryClass, IntentClass]] = None
    number_rows: int = Field(default=3, le=7)
                             
class SummarizeText(BaseModel):
    function_type: Literal["summarize_text"]
    text_field: Optional[UnstructedField] = None
    condition_field: Optional[StructedField] = None
    condition_field_value: Optional[Union[CategoryClass, IntentClass]] = None
    number_rows: int = 10

class Finish(BaseModel):
    function_type: Literal["finish"]

# Discriminated union for function inputs

FunctionType = Union[
                     DatasetOverview, 
                     SelectSemanticIntent, 
                     SelectSemanticCategory, 
                     CountIntent, CountCategory, 
                     SumValues, 
                     MultiplicationFloat, 
                     DivisionFloat, 
                     ShowExamples, 
                     SummarizeText, 
                     Finish
                     ]

class FunctionInput(BaseModel):
    function_call: FunctionType = Field(discriminator="function_type")