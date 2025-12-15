#!/usr/bin/env python3
"""
Content validation utilities for security and safety checks.

This module provides comprehensive validation for URLs and content types
to prevent security risks when processing unknown domains.
"""

import re
from typing import NamedTuple

import requests

from .config import get_config
from .piedomains_logging import get_logger

logger = get_logger()


class ContentValidationResult(NamedTuple):
    """Result of content validation check."""

    is_safe: bool
    content_type: str | None
    content_length: int | None
    error_message: str
    warnings: list[str]
    sandbox_recommended: bool


class SecurityWarning(Exception):
    """Exception raised for security-related content validation failures."""

    pass


class ContentValidator:
    """Validates content safety and provides security recommendations."""

    def __init__(self, config=None):
        """Initialize content validator with configuration."""
        self.config = config or get_config()

    def validate_url(
        self,
        url: str,
        *,
        force_fetch: bool = False,
        allow_content_types: list[str] | None = None,
        ignore_extensions: bool = False,
    ) -> ContentValidationResult:
        """
        Comprehensive URL and content validation.

        Args:
            url: URL to validate
            force_fetch: Skip validation and force content retrieval
            allow_content_types: Override allowed content types
            ignore_extensions: Skip file extension validation

        Returns:
            ContentValidationResult with validation details
        """
        warnings = []

        # Skip validation if disabled or force_fetch is True
        if not self.config.enable_content_validation or force_fetch:
            if force_fetch:
                warnings.append("Content validation bypassed by user request")
            return ContentValidationResult(
                is_safe=True,
                content_type=None,
                content_length=None,
                error_message="",
                warnings=warnings,
                sandbox_recommended=False,
            )

        # Step 1: URL pattern validation
        url_validation = self._validate_url_patterns(url, ignore_extensions)
        if not url_validation.is_safe:
            return url_validation

        # Step 2: Pre-flight HEAD request to check content type
        try:
            preflight_result = self._preflight_check(url, allow_content_types)
            if not preflight_result.is_safe:
                return preflight_result

            warnings.extend(preflight_result.warnings)

            return ContentValidationResult(
                is_safe=True,
                content_type=preflight_result.content_type,
                content_length=preflight_result.content_length,
                error_message="",
                warnings=warnings,
                sandbox_recommended=preflight_result.sandbox_recommended,
            )

        except Exception as e:
            logger.warning(f"Pre-flight validation failed for {url}: {e}")
            # If pre-flight fails, still allow but with warning
            warnings.append(f"Pre-flight check failed: {e}")
            return ContentValidationResult(
                is_safe=True,
                content_type=None,
                content_length=None,
                error_message="",
                warnings=warnings,
                sandbox_recommended=True,  # Recommend sandbox on uncertainty
            )

    def _validate_url_patterns(
        self, url: str, ignore_extensions: bool
    ) -> ContentValidationResult:
        """Validate URL patterns for suspicious content."""
        warnings = []

        # Check for dangerous file extensions, but distinguish between domains and file paths
        if not ignore_extensions:
            # Common domain TLDs that should not be blocked
            domain_tlds = {
                ".com",
                ".org",
                ".net",
                ".edu",
                ".gov",
                ".mil",
                ".int",
                ".info",
                ".biz",
                ".name",
                ".pro",
                ".museum",
                ".coop",
                ".aero",
                ".jobs",
                ".mobi",
                ".travel",
                ".tel",
                ".asia",
                ".cat",
                ".post",
                ".xxx",
            }

            # Parse URL to distinguish domain from path
            from urllib.parse import urlparse

            parsed = urlparse(url if "://" in url else f"http://{url}")
            domain = parsed.netloc.lower()
            path = parsed.path.lower()

            # Only validate file extensions in the path, not the domain
            for ext in self.config.blocked_extensions:
                ext_lower = ext.lower()

                # Skip if this is just a domain ending in a TLD (unless explicitly configured to validate domains)
                if (
                    path in ("", "/")
                    and any(domain.endswith(tld) for tld in domain_tlds)
                    and not getattr(self.config, "validate_domain_extensions", False)
                ):
                    continue

                # Check for dangerous extensions in the path
                if (
                    path.endswith(ext_lower)
                    or f"{ext_lower}?" in path
                    or (
                        path == ""
                        and url.lower().endswith(ext_lower)
                        and not any(url.lower().endswith(tld) for tld in domain_tlds)
                    )
                ):
                    error_msg = (
                        f"Blocked file extension detected: {ext}\n"
                        f"This appears to be a binary/executable file.\n"
                        f"Recommendation: Use sandbox execution for safety.\n"
                        f"Example: python3 examples/sandbox/secure_classify.py '{url}'"
                    )
                    return ContentValidationResult(
                        is_safe=False,
                        content_type=None,
                        content_length=None,
                        error_message=error_msg,
                        warnings=warnings,
                        sandbox_recommended=True,
                    )

        # Check suspicious URL patterns
        for pattern in self.config.suspicious_url_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                error_msg = (
                    f"Suspicious URL pattern detected: {pattern}\n"
                    f"This URL suggests binary/downloadable content.\n"
                    f"Recommendation: Use sandbox execution for safety.\n"
                    f"Example: python3 examples/sandbox/secure_classify.py '{url}'"
                )
                return ContentValidationResult(
                    is_safe=False,
                    content_type=None,
                    content_length=None,
                    error_message=error_msg,
                    warnings=warnings,
                    sandbox_recommended=True,
                )

        return ContentValidationResult(
            is_safe=True,
            content_type=None,
            content_length=None,
            error_message="",
            warnings=warnings,
            sandbox_recommended=False,
        )

    def _preflight_check(
        self, url: str, allow_content_types: list[str] | None = None
    ) -> ContentValidationResult:
        """Perform HEAD request to validate content before download."""
        warnings = []

        try:
            # Make HEAD request to get headers without downloading content
            response = requests.head(
                url,
                timeout=self.config.http_timeout,
                headers={"User-Agent": self.config.user_agent},
                allow_redirects=True,
            )
            response.raise_for_status()

        except requests.RequestException as e:
            # If HEAD fails, try GET with range header (some servers don't support HEAD)
            try:
                response = requests.get(
                    url,
                    timeout=self.config.http_timeout,
                    headers={
                        "User-Agent": self.config.user_agent,
                        "Range": "bytes=0-1023",  # Just get first 1KB
                    },
                    stream=True,
                    allow_redirects=True,
                )
                response.raise_for_status()
            except requests.RequestException:
                raise e from None  # Re-raise original HEAD error

        # Extract content information
        content_type = (
            response.headers.get("content-type", "").split(";")[0].strip().lower()
        )
        content_length = self._parse_content_length(
            response.headers.get("content-length")
        )
        content_disposition = response.headers.get("content-disposition", "")

        # Check for forced downloads
        if "attachment" in content_disposition.lower():
            error_msg = (
                f"Content-Disposition indicates forced download.\n"
                f"This suggests binary/downloadable content rather than a webpage.\n"
                f"Content-Type: {content_type}\n"
                f"Recommendation: Use sandbox execution for safety.\n"
                f"Example: python3 examples/sandbox/secure_classify.py '{url}'"
            )
            return ContentValidationResult(
                is_safe=False,
                content_type=content_type,
                content_length=content_length,
                error_message=error_msg,
                warnings=warnings,
                sandbox_recommended=True,
            )

        # Validate content type
        allowed_types = allow_content_types or self.config.allowed_content_types
        safety_mode = self.config.content_safety_mode

        if content_type:
            # Check if content type is explicitly allowed
            is_allowed_type = any(
                content_type.startswith(allowed.lower()) for allowed in allowed_types
            )

            if not is_allowed_type:
                # Handle based on safety mode
                if safety_mode == "strict":
                    error_msg = self._generate_content_type_error(url, content_type)
                    return ContentValidationResult(
                        is_safe=False,
                        content_type=content_type,
                        content_length=content_length,
                        error_message=error_msg,
                        warnings=warnings,
                        sandbox_recommended=True,
                    )
                elif safety_mode == "moderate":
                    warnings.append(f"Potentially unsafe content type: {content_type}")
                    # Continue but recommend sandbox for certain types
                    sandbox_recommended = self._should_recommend_sandbox(content_type)
                else:  # permissive
                    warnings.append(f"Non-standard content type: {content_type}")
                    sandbox_recommended = False
            else:
                sandbox_recommended = False
        else:
            # No content type header
            warnings.append("No Content-Type header found")
            sandbox_recommended = True

        # Validate content length
        length_validation = self._validate_content_length(content_type, content_length)
        if not length_validation.is_safe:
            return length_validation

        warnings.extend(length_validation.warnings)

        return ContentValidationResult(
            is_safe=True,
            content_type=content_type,
            content_length=content_length,
            error_message="",
            warnings=warnings,
            sandbox_recommended=sandbox_recommended,
        )

    def _parse_content_length(self, content_length_header: str | None) -> int | None:
        """Parse Content-Length header safely."""
        if not content_length_header:
            return None
        try:
            return int(content_length_header)
        except ValueError:
            return None

    def _validate_content_length(
        self, content_type: str | None, content_length: int | None
    ) -> ContentValidationResult:
        """Validate content length against limits."""
        warnings = []

        if content_length is None:
            warnings.append("Content-Length not available - cannot verify size")
            return ContentValidationResult(
                is_safe=True,
                content_type=content_type,
                content_length=None,
                error_message="",
                warnings=warnings,
                sandbox_recommended=False,
            )

        # Get appropriate limit
        limits = self.config.content_length_limits
        limit = limits.get(content_type or "", limits["default"])

        if content_length > limit:
            size_mb = content_length / (1024 * 1024)
            limit_mb = limit / (1024 * 1024)
            error_msg = (
                f"Content too large: {size_mb:.1f}MB (limit: {limit_mb:.1f}MB)\n"
                f"Large files may consume excessive resources or contain binary data.\n"
                f"Content-Type: {content_type}\n"
                f"Recommendation: Use text-only analysis or sandbox execution.\n"
                f"Example: python3 examples/sandbox/secure_classify.py '{content_type}' --text-only"
            )
            return ContentValidationResult(
                is_safe=False,
                content_type=content_type,
                content_length=content_length,
                error_message=error_msg,
                warnings=warnings,
                sandbox_recommended=True,
            )

        return ContentValidationResult(
            is_safe=True,
            content_type=content_type,
            content_length=content_length,
            error_message="",
            warnings=warnings,
            sandbox_recommended=False,
        )

    def _generate_content_type_error(self, url: str, content_type: str) -> str:
        """Generate user-friendly error message for unsupported content types."""

        # Specific guidance for common problematic types
        if content_type.startswith("application/pdf"):
            return (
                f"PDF document detected (Content-Type: {content_type})\n"
                f"PDFs cannot be processed as webpages and may contain malicious content.\n"
                f"Recommendations:\n"
                f"• Use PDF text extraction tools instead\n"
                f"• For safety analysis, use sandbox: python3 examples/sandbox/secure_classify.py '{url}'\n"
                f"• Consider text-only analysis if this is actually a webpage"
            )
        elif content_type.startswith("application/octet-stream"):
            return (
                f"Binary content detected (Content-Type: {content_type})\n"
                f"This appears to be a binary file rather than a webpage.\n"
                f"Recommendations:\n"
                f"• Verify the URL is correct\n"
                f"• For safety analysis, use sandbox: python3 examples/sandbox/secure_classify.py '{url}'\n"
                f"• Use --force-fetch to override this check (NOT recommended for unknown sources)"
            )
        elif content_type.startswith("application/zip") or "archive" in content_type:
            return (
                f"Archive file detected (Content-Type: {content_type})\n"
                f"Archive files cannot be processed as webpages and may contain malicious content.\n"
                f"Recommendations:\n"
                f"• Verify the URL is correct\n"
                f"• For safety analysis, use sandbox: python3 examples/sandbox/secure_classify.py '{url}'\n"
                f"• Use --force-fetch to override this check (NOT recommended)"
            )
        else:
            return (
                f"Unsupported content type: {content_type}\n"
                f"This content type is not safe for webpage analysis.\n"
                f"Allowed types: {', '.join(self.config.allowed_content_types)}\n"
                f"Recommendations:\n"
                f"• Verify the URL points to a webpage\n"
                f"• For safety analysis, use sandbox: python3 examples/sandbox/secure_classify.py '{url}'\n"
                f"• Use --allow-content-types=['{content_type}'] to override\n"
                f"• Use --force-fetch to bypass all checks (NOT recommended for unknown sources)"
            )

    def _should_recommend_sandbox(self, content_type: str) -> bool:
        """Determine if sandbox should be recommended for content type."""
        risky_types = [
            "application/pdf",
            "application/octet-stream",
            "application/zip",
            "application/x-executable",
            "application/x-msdownload",
            "application/x-msdos-program",
        ]

        return any(content_type.startswith(risky) for risky in risky_types)

    def get_sandbox_command(self, url: str, text_only: bool = True) -> str:
        """Generate sandbox execution command for a URL."""
        script_path = "examples/sandbox/secure_classify.py"
        options = "--text-only" if text_only else ""
        return f"python3 {script_path} '{url}' {options}".strip()
