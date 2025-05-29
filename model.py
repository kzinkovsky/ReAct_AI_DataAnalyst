import enum
from typing import Optional, Union, List, Literal
from pydantic import BaseModel, Field

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

class CountOperation(str, enum.Enum):
    LIST_UNIQUE = 'list_unique'
    UNIQUE_COUNT = 'unique_count'
    DISRIBUTION = 'distribution'
    CLASS_COUNT = 'class_count'
    MOST_COMMON = 'most_common'
    LEAST_FREQUENT = 'least_frequent'

# Pydantic models for function inputs

class RetrieveTextInput(BaseModel):
    function_type: Literal["retrieve_text"]
    target_field: Optional[UnstructedField] = None
    condition_field: Optional[StructedField] = None
    condition_class: Optional[Union[CategoryClass, IntentClass]] = None
    number_rows: int = 1

class SummarizeTextInput(BaseModel):
    function_type: Literal["summarize_text"]
    text_list: List[str]

class CountStructuredInput(BaseModel):
    function_type: Literal["count_structured"]
    target_field: StructedField
    count_operation: CountOperation
    target_class: Optional[Union[CategoryClass, IntentClass]] = None

# Discriminated union for function inputs

FunctionType = Union[RetrieveTextInput, SummarizeTextInput, CountStructuredInput]

class FunctionInput(BaseModel):
    function_call: FunctionType = Field(discriminator="function_type")