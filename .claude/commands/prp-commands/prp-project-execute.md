# Execute Project PRP

Executes a multi-stage project by following the chain of PRPs created by `prp-project-create`, implementing each stage in dependency order with validation and integration testing.

## Arguments: $ARGUMENTS

This command takes a path to a master project PRP file and executes the entire project by:
1. Loading the project plan and dependency graph
2. Executing each stage PRP in topological order
3. Validating stage completion before proceeding
4. Performing integration testing between stages
5. Conducting end-to-end system validation

## Core Process

### Stage 1: Project Plan Loading & Validation

Begin by loading and validating the project execution plan:

1. **Load Master Project PRP**:
   ```
   Read the master project PRP file specified in $ARGUMENTS
   Parse the project overview, stage definitions, and dependency graph
   Validate that all referenced stage PRP files exist
   ```

2. **Dependency Graph Analysis**:
   ```
   Build a topological ordering of all project stages
   Identify the critical path for project execution
   Check for circular dependencies or invalid stage references
   Determine parallel execution opportunities (if any)
   ```

3. **Pre-Execution Validation**:
   ```
   Verify all stage PRPs are properly formatted and complete
   Check that inter-stage dependencies are clearly defined
   Validate that required project context and patterns are available
   Ensure all necessary tools and environments are accessible
   ```

### Stage 2: Sequential Stage Execution

Execute each stage following the dependency order:

1. **Stage Preparation**:
   - Load the stage PRP file
   - Prepare the execution context with:
     - Outputs from completed previous stages
     - Global project patterns and conventions
     - Inter-stage interface specifications
     - Stage-specific requirements and constraints

2. **Stage Implementation**:
   - Execute the stage PRP using the standard `execute-base-prp` logic
   - Monitor implementation progress and handle errors
   - Apply stage-specific validation loops
   - Ensure all stage success criteria are met

3. **Stage Validation**:
   - Run stage-specific unit and integration tests
   - Validate outputs match expected interface specifications  
   - Verify compatibility with next stages in the dependency chain
   - Update stage status and document any issues or deviations

4. **Inter-Stage Integration**:
   - Test integration points with previously completed stages
   - Validate data flows and interface compatibility
   - Run integration tests spanning multiple completed stages
   - Update project-wide integration status

### Stage 3: Progress Monitoring & Error Handling

Maintain project execution state and handle issues:

1. **Execution State Tracking**:
   ```
   Track completion status of each stage
   Maintain execution logs and progress reports
   Record integration test results and validation outcomes
   Monitor project-wide health and dependency satisfaction
   ```

2. **Error Recovery**:
   ```
   On stage failure:
   - Pause execution and analyze failure cause
   - Determine if failure affects downstream stages
   - Provide detailed error reporting and suggested fixes
   - Allow for manual intervention and stage retry
   ```

3. **Quality Gates**:
   ```
   Before proceeding to next stage:
   - All stage tests must pass
   - Integration points must be validated
   - Success criteria must be completely satisfied
   - No critical issues or technical debt introduced
   ```

### Stage 4: Project Completion & Validation

Finalize the project with comprehensive validation:

1. **End-to-End Testing**:
   ```
   Run comprehensive system-wide tests
   Validate all integration points work correctly
   Test complete user workflows across all stages
   Verify project meets original success criteria
   ```

2. **Documentation & Handoff**:
   ```
   Generate project completion report
   Document architecture decisions and patterns used
   Create deployment and maintenance guides
   Update project documentation with final specifications
   ```

## Implementation Details

Execute this command following these detailed steps:

1. **Initialize Execution Environment**:
   ```bash
   # Parse project PRP file path from arguments
   PROJECT_PRP_PATH="$ARGUMENTS"
   
   # Validate file exists and is readable
   if [[ ! -f "$PROJECT_PRP_PATH" ]]; then
       echo "Error: Project PRP file not found: $PROJECT_PRP_PATH"
       exit 1
   fi
   
   # Create execution tracking directory
   mkdir -p .prp-execution/$(basename "$PROJECT_PRP_PATH" .md)
   EXECUTION_LOG_DIR=".prp-execution/$(basename "$PROJECT_PRP_PATH" .md)"
   ```

2. **Parse Project Structure**:
   ```
   Extract stage list and dependency information from master PRP
   Build execution order based on dependency graph
   Validate all stage PRP files are accessible
   Initialize execution tracking and logging
   ```

3. **Execute Stage Chain**:
   ```
   For each stage in dependency order:
     1. Log stage start and load stage PRP
     2. Prepare stage context with previous outputs
     3. Execute stage using /execute-base-prp logic
     4. Run stage validation and integration tests
     5. Update execution status and logs
     6. Proceed to next stage if all validations pass
   ```

4. **Project Completion**:
   ```
   Run end-to-end system tests
   Generate execution summary report  
   Mark project as completed if all stages successful
   Provide next steps and deployment guidance
   ```

## Execution Modes

Support different execution modes for various use cases:

### Interactive Mode (Default)
- Pause after each stage for user review and approval
- Allow manual intervention and debugging
- Provide detailed progress feedback and recommendations

### Headless Mode
- Execute all stages automatically without user intervention
- Suitable for CI/CD pipeline integration
- Provide comprehensive logging and error reporting

### Resume Mode
- Resume execution from the last completed stage
- Handle partial execution scenarios
- Maintain execution state across sessions

## File Structure During Execution

```
project-root/
├── PRPs/
|   └── [name]
│        ├── project-[name].md                    # Master project PRP
│        ├── stage-1-[name].md                   # Individual stage PRPs
│        ├── stage-2-[name].md
│        └── stage-N-[name].md
├── .prp-execution/
│   └── project-[name]/
│       ├── execution.log                   # Detailed execution log
│       ├── stage-status.json              # Stage completion status
│       ├── integration-results.json       # Integration test results
│       └── stage-outputs/                 # Stage output artifacts
│           ├── stage-1/
│           ├── stage-2/
│           └── stage-N/
└── [project implementation files]
```

## Success Criteria

The project execution succeeds when:
- [ ] All stages complete successfully in dependency order
- [ ] All stage-specific tests pass
- [ ] All inter-stage integration tests pass
- [ ] End-to-end system tests pass
- [ ] All original project success criteria are met
- [ ] No critical issues or blocking technical debt remains
- [ ] Complete execution log and documentation generated

## Error Handling Strategies

1. **Stage Execution Failures**:
   - Log detailed error information
   - Analyze impact on downstream stages
   - Provide specific remediation suggestions
   - Allow for stage retry after fixes

2. **Integration Test Failures**:
   - Identify specific integration points failing
   - Suggest interface or data flow corrections
   - Allow targeted re-execution of affected stages
   - Update integration specifications if needed

3. **Dependency Resolution Issues**:
   - Identify circular or missing dependencies
   - Suggest project structure corrections
   - Allow manual dependency override if safe
   - Update master project PRP if needed

## Integration with Existing PRP Workflow

This command seamlessly integrates with the existing PRP ecosystem:

- Uses `execute-base-prp` logic for individual stage execution
- Follows standard PRP validation loop patterns
- Maintains compatibility with existing PRP templates
- Leverages established context engineering practices
- Integrates with existing debugging and review workflows

## Monitoring and Reporting

Provide comprehensive project execution visibility:

1. **Real-time Progress Updates**:
   - Current stage being executed
   - Completion percentage for overall project
   - Estimated time remaining based on stage complexity
   - Active issues or blockers requiring attention

2. **Execution Summary Reports**:
   - Total execution time per stage and overall
   - Test results and validation outcomes
   - Integration points successfully established
   - Technical debt or issues identified during execution

3. **Post-Execution Analysis**:
   - Stage execution efficiency metrics
   - Dependency resolution accuracy
   - Integration complexity analysis
   - Recommendations for future similar projects

The goal is to provide a robust, automated project execution engine that can handle complex multi-stage projects while maintaining the quality and validation standards of the PRP methodology.