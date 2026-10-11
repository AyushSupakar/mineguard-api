from django.urls import path
from .views import AuditFrameView, CopilotQueryView

urlpatterns = [
    path('audit/', AuditFrameView.as_view(), name='audit-frame'),
    path('copilot/', CopilotQueryView.as_view(), name='statutory-copilot'),
]